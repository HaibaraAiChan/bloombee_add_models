from petals.models.opt.block import WrappedOPTBlock  
from petals.models.opt.config import DistributedOPTConfig  
from petals.models.opt.model import (  
    DistributedOPTForCausalLM,  
    DistributedOPTForSequenceClassification,  
    DistributedOPTModel,  
)  
# from petals.models.opt.speculative_model import DistributedOPTForSpeculativeGeneration  
from petals.utils.auto_config import register_model_classes  

register_model_classes(  
    config=DistributedOPTConfig,  
    model=DistributedOPTModel,  
    model_for_causal_lm=DistributedOPTForCausalLM,  
    # model_for_speculative=DistributedOPTForSpeculativeGeneration,  
    model_for_sequence_classification=DistributedOPTForSequenceClassification,  
)