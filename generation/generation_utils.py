from transformers import AutoConfig

# get a language model (for perplexity)
def is_seq2seq(model_or_model_name_or_path):
    config = get_config(model_or_model_name_or_path)
    return config.is_encoder_decoder is True

def is_perplexity(model_or_model_name_or_path):
    config = get_config(model_or_model_name_or_path)
    return config.is_encoder_decoder or config.is_decoder

def get_config(model_or_model_name_or_path):
    if isinstance(model_or_model_name_or_path, str):
        config = AutoConfig.from_pretrained(model_or_model_name_or_path, trust_remote_code=True)
    else:
        config = model_or_model_name_or_path.model.config
    return config

