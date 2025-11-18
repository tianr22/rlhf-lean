from torch.utils.data import Dataset
from typing import Callable
from openrlhf.utils.utils import zero_pad_sequences
import torch

class ValueDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.prompt_key = getattr(self.strategy.args,"prompt_key", None)
        self.value_key = getattr(self.strategy.args,"value_key", None)

        self.prompts = dataset[self.prompt_key]
        self.values = dataset[self.value_key]

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self,idx):
        tokenizer = self.tokenizer
        prompt = self.prompts[idx]
        value  = self.values[idx]

        prompt_chat = [{"role": "user", "content": prompt}]

        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize = False)

        prompt_ids = tokenizer(
            prompt_chat_str,
            max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False,
        )

        prompt_ids["input_ids"][0][-1] = self.tokenizer.eos_token_id
        prompt_ids["attention_mask"][0][-1] = True

        return (
            prompt_ids["input_ids"],
            prompt_ids["attention_mask"],
            value,
        )
        
    def collate_fn(self, item_list):
        prompt_ids = []
        prompt_masks = []
        values = []

        for prompt_id, prompt_mask, value in item_list:
            prompt_ids.append(prompt_id)
            prompt_masks.append(prompt_mask)
            values.append(value)

        padding_side = "left"

        prompt_ids = zero_pad_sequences(
            prompt_ids,
            side = padding_side,
            value = self.tokenizer.pad_token_id,
        )
        prompt_masks = zero_pad_sequences(
            prompt_masks,
            side = padding_side,
        )
    
        values = torch.tensor(values, dtype=torch.float32)  

        return prompt_ids, prompt_masks, values