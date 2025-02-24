from transformers import AutoConfig  

# Specify the model name  
model_name = "model-attribution-challenge/bloom-350m"  

# Load the configuration  
config = AutoConfig.from_pretrained(model_name)  

# Print the configuration  
print(config)
