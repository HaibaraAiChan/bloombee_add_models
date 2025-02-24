import os  
from typing import Optional, Union  

from hivemind import get_logger  
from transformers.models.opt import OPTConfig
from transformers.models.bloom import BloomConfig
from transformers.models.opt.modeling_opt import OPTAttention  
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.client.config import ClientConfig  
from petals.client.lm_head import LMHeadConfig  
from petals.client.ptune import PTuneConfig  
from petals.models.opt.block import WrappedOPTBlock  

logger = get_logger(__name__)  


class DistributedOPTConfig(OPTConfig, BloomConfig, ClientConfig, PTuneConfig, LMHeadConfig):  
    block_class = WrappedOPTBlock  
    attn_class = BloomAttention  
    block_prefix = "h"
    # block_prefix = "model.decoder.layers"  
    
    num_key_value_groups = 1
    # layer_norm_epsilon=1e-5
    # n_head=8
    hidden_size=1024
    
    @classmethod  
    def from_pretrained(  
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs  
    ):  
        logger.info(  
            "Make sure you follow the OPT terms of use: "  
            "https://github.com/facebookresearch/fairseq/blob/main/LICENSE"  
        )  

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)  
        if loading_from_repo and dht_prefix is None:  
            dht_prefix = str(model_name_or_path)  
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts  
            dht_prefix = dht_prefix.replace(".", "-")  
            if not dht_prefix.endswith("-hf"):  
                dht_prefix += "-hf"  
            logger.info(f"Using DHT prefix: {dht_prefix}")  
        # import pdb;pdb.set_trace()
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        # print('result ',result) is not correct 
        if isinstance(result, tuple):   # correct 1024
            result[0].hidden_size = 1024  
        else:  
            result.hidden_size = 1024
        # print('after from_pretrained() result, ', result)  
        config = result[0] if isinstance(result, tuple) else result  
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Petals  
        # print('')
        
        return result