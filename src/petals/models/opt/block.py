"""  
OPT intermediate layer  
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py  
See commit history for authorship.  
"""  
import math  
from typing import Optional, Tuple  

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask  
from transformers.models.opt.modeling_opt import (  
    OPTAttention,  
    OPTConfig,  
    OPTDecoderLayer,  
)  

from petals.utils.cuda_graphs import make_inference_graphed_callable  


class LayerNorm(nn.Module):  
    def __init__(self, hidden_size, eps=1e-5):  
        super().__init__()  
        self.weight = nn.Parameter(torch.ones(hidden_size))  
        self.bias = nn.Parameter(torch.zeros(hidden_size))  
        self.variance_epsilon = eps  

    def forward(self, hidden_states):  
        input_type = hidden_states.dtype  
        hidden_states = hidden_states.float()  
        mean = hidden_states.mean(-1, keepdim=True)  
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)  
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)  
        hidden_states = hidden_states.to(input_type)  
        return self.weight * hidden_states + self.bias  


class OPTMLP(nn.Module):  
    def __init__(self, config: OPTConfig):  
        super().__init__()  
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size*4)  
        self.fc2 = nn.Linear(config.hidden_size*4, config.hidden_size)  
        self.activation = F.gelu  

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  
        hidden_states = self.fc1(hidden_states)  
        hidden_states = self.activation(hidden_states)  
        hidden_states = self.fc2(hidden_states)  
        return hidden_states  


class OptimizedOPTAttention(OPTAttention):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        self._attention_graph = None  

    def _optimized_attention(self, query_states, key_states, value_states, attention_mask):  
        if self._attention_graph is None:  
            self._attention_graph = make_inference_graphed_callable(  
                self._attention_forward, sample_args=(query_states, key_states, value_states, attention_mask)  
            )  
        return self._attention_graph(query_states, key_states, value_states, attention_mask)  

    def _attention_forward(self, query_states, key_states, value_states, attention_mask):  
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)  
        if attention_mask is not None:  
            attn_weights = attn_weights + attention_mask  
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  
        attn_output = torch.matmul(attn_weights, value_states)  
        return attn_output  

    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.LongTensor] = None,  
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  
        output_attentions: bool = False,  
        use_cache: bool = False,  
        cache_position: Optional[torch.LongTensor] = None,  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:  
        assert not output_attentions  
        if position_ids is None:  
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0  
            position_ids = torch.arange(  
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device  
            ).unsqueeze(0)  

        bsz, q_len, _ = hidden_states.size()  

        query_states = self.q_proj(hidden_states)  
        key_states = self.k_proj(hidden_states)  
        value_states = self.v_proj(hidden_states)  

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  

        if past_key_value is not None:  
            key_states = torch.cat([past_key_value[0], key_states], dim=2)  
            value_states = torch.cat([past_key_value[1], value_states], dim=2)  

        past_key_value = (key_states, value_states) if use_cache else None  

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":  
            attn_output = self._optimized_attention(query_states, key_states, value_states, attention_mask)  
        else:  
            attn_output = self._attention_forward(query_states, key_states, value_states, attention_mask)  

        attn_output = attn_output.transpose(1, 2).contiguous()  
        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)  

        attn_output = self.out_proj(attn_output)  

        return attn_output, None, past_key_value  


class OptimizedOPTDecoderLayer(OPTDecoderLayer):  
    def __init__(self, config: OPTConfig):  
        nn.Module.__init__(self)  
        self.embed_dim = config.hidden_size  
        self.self_attn = OptimizedOPTAttention(config=config, is_decoder=True)  
        self.mlp = OPTMLP(config)  
        self.input_layernorm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)  
        self.post_attention_layernorm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)  

        self.pre_attn_graph = None  
        self.post_attn_graph = None  

    def _optimized_input_layernorm(self, hidden_states):  
        if self.pre_attn_graph is None:  
            self.pre_attn_graph = make_inference_graphed_callable(  
                self.input_layernorm.forward, sample_args=(hidden_states,)  
            )  
        return self.pre_attn_graph(hidden_states)  

    def _optimized_output_layernorm(self, hidden_states):  
        if self.post_attn_graph is None:  
            self.post_attn_graph = make_inference_graphed_callable(  
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)  
            )  
        return self.post_attn_graph(hidden_states)  

    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.LongTensor] = None,  
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  
        output_attentions: Optional[bool] = False,  
        use_cache: Optional[bool] = False,  
        cache_position: Optional[torch.LongTensor] = None,  
        **kwargs,  
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:  
        residual = hidden_states  

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":  
            hidden_states = self._optimized_input_layernorm(hidden_states)  
        else:  
            hidden_states = self.input_layernorm(hidden_states)  

        # Self Attention  
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  
            hidden_states=hidden_states,  
            attention_mask=attention_mask,  
            position_ids=position_ids,  
            past_key_value=past_key_value,  
            output_attentions=output_attentions,  
            use_cache=use_cache,  
            cache_position=cache_position,  
            **kwargs,  
        )  

        hidden_states = residual + hidden_states  

        # Fully Connected  
        residual = hidden_states  

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":  
            hidden_states = self._optimized_output_layernorm(hidden_states)  
        else:  
            hidden_states = self.post_attention_layernorm(hidden_states)  

        hidden_states = self.mlp(hidden_states)  
        hidden_states = residual + hidden_states  

        outputs = (hidden_states,)  

        if output_attentions:  
            outputs += (self_attn_weights,)  

        if use_cache:  
            outputs += (present_key_value,)  

        return outputs  


class WrappedOPTBlock(OptimizedOPTDecoderLayer):  
    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        *args,  
        attention_mask: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.LongTensor] = None,  
        layer_past: Optional[Tuple[torch.Tensor]] = None,  
        use_cache: bool = False,  
        **kwargs,  
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:  
        batch_size, seq_length, _ = hidden_states.shape  

        seq_length_with_past = seq_length  
        past_key_values_length = 0  

        past_key_value = layer_past  
        if past_key_value is not None:  
            past_key_values_length = past_key_value[0].shape[2]  
            seq_length_with_past = seq_length_with_past + past_key_values_length  
            past_key_value = self._reorder_cache_from_bloom_to_opt(past_key_value, batch_size, past_key_values_length)  

        assert position_ids is None  

        # embed positions  
        if attention_mask is None:  
            attention_mask = torch.ones(  
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device  
            )  
        attention_mask = _prepare_4d_causal_attention_mask(  
            attention_mask=attention_mask,  
            input_shape=(batch_size, seq_length),  
            inputs_embeds=hidden_states,  
            past_key_values_length=past_key_values_length,  
        )  

        outputs = super().forward(  
            hidden_states,  
            *args,  
            attention_mask=attention_mask,  
            position_ids=position_ids,  
            past_key_value=past_key_value,  
            use_cache=use_cache,  
            **kwargs,  
        )  

        if use_cache:  
            present_key_value = outputs[-1]  
            present_key_value = self._reorder_cache_from_opt_to_bloom(  
                present_key_value, batch_size, seq_length_with_past  
            )  
            outputs = outputs[:-1] + (present_key_value,)  

        return outputs  

    def _reorder_cache_from_bloom_to_opt(  
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int  
    ) -> Tuple[torch.Tensor]:  
        key_states, value_states = key_value  
        key_states = key_states.permute(0, 2, 1)  
        key_states = key_states.view(  
            batch_size, self.self_attn.num_heads, seq_length, self.self_attn.head_dim  
        )  
        value_states = value_states.view(*key_states.shape)  
        return (key_states, value_states)  

    def _reorder_cache_from_opt_to_bloom(  
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int  
    ) -> Tuple[torch.Tensor]:  
        key_states, value_states = key_value  
        value_states = value_states.view(  
            batch_size * self.self_attn.num_heads, seq_length, self.self_attn.head_dim  
        )  
        key_states = key_states.view(*value_states.shape)  
        key_states = key_states.permute(0, 2, 1)  
        return (key_states, value_states)