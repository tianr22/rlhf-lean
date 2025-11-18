from transformers import AutoModelForSequenceClassification
import torch

reward_model_path = "/data2/tianrui/models"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
)


# save the model
save_path = "/data2/tianrui/reward_model"
reward_model.save_pretrained(save_path)
print(f"Reward model saved to {save_path}")
