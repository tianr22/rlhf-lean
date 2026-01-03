from typing import Callable

from torch.utils.data import Dataset
from openrlhf.lean.Package import Package
from openrlhf.utils.utils import zero_pad_sequences

    
class OnlineRewardDatasetLean(Dataset):
    def __init__(
        self,
        data,
        tokenizer: Callable,
        max_length: int,
        strategy,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
        self.chosens, self.rejects = self.process_data(data['chosen'], data['rejected'])

    def process_data(self, chosens, rejects):
        processed_chosens = []
        processed_rejects = []
        for chosen, rejected in zip(chosens, rejects):
            chosen = [{"role": "user", "content": chosen}]
            rejected = [{"role": "user", "content": rejected}]
            chosen = self.apply_chat_template(chosen, tokenize=False)
            rejected = self.apply_chat_template(rejected, tokenize=False)
            processed_chosens.append(chosen)
            processed_rejects.append(rejected)
        return processed_chosens, processed_rejects

    def __len__(self):
        return len(self.chosens)
    
    def __getitem__(self, idx):
        chosen, reject =  self.chosens[idx], self.rejects[idx]

        chosen = chosen.rstrip("\n")
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

        reject = reject.rstrip("\n")
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
        )
    
    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        for chosen_id, chosen_mask, reject_id, rejects_mask in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)

        padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks

    
class PackageDataset(Dataset):
    def __init__(
        self,
        dataset,
        num_processors=8,
    ) -> None:
        super().__init__()
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        self.packages = processed_dataset["package"]

    def process_data(self,data):
        package = Package.from_dict(data)

        return {
            "package": package,
        }
    
    def __len__(self):
        return len(self.packages)
    
    def __getitem__(self, index):
        package = self.packages[index]
        return (
            package,
        )

    def collate_fn(self, item_list):
        packages = []
        for package in item_list:
            packages.append(package)

        return packages