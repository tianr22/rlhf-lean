from typing import Callable

from torch.utils.data import Dataset
from openrlhf.utils.utils import zero_pad_sequences

def preprocess_data(
    data,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
) -> str:
        prompt = ""
        chosen = [{"role": "user", "content": data[chosen_key]}]
        rejected = [{"role": "user", "content": data[rejected_key]}]
        chosen = apply_chat_template(chosen, tokenize=False)
        rejected = apply_chat_template(rejected, tokenize=False)

        # margin loss
        margin = 0

        return prompt, chosen, rejected, margin

class RewardDatasetLean(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)

        self.chosens = dataset[self.chosen_key]
        self.rejects = dataset[self.rejected_key]

        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template

        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": margin,
        }
    
    def __len__(self):
        length = len(self.chosens)
        return length   

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )
    
    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
