import torch
from typing import List

from openrlhf.lean.Package import Package, Trial
from openrlhf.utils.utils import zero_pad_sequences
import random

class LEANSamplingTree:
    def __init__(self, 
        tokenizer, 
        model_type,
        value_model,
        req_agent,
        lean_verifier,
        max_length,
        max_trails_per_package=32
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.value_model = value_model
        self.req_agent = req_agent
        self.lean_verifier = lean_verifier  
        self.max_length = max_length
        self.max_trails_per_package = max_trails_per_package
    
    def tokenize_value_prompt(self, prompt):
            value_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
            ).rstrip("\n")
            if not value_prompt.endswith(self.tokenizer.eos_token):
                value_prompt += " " + self.tokenizer.eos_token
            value_token = self.tokenizer(
                value_prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            value_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            value_token["attention_mask"][0][-1] = True
            return value_token

    def generate_value_pair(self, packages: List[Package]):
        value_pair = {'chosen': [], 'reject': []}
        for package in packages:
            if not package.passed:
                continue
            
            pair = []
            # find the sussessful trial
            success_idx = -1
            for i, trial in enumerate(package.trials):
                if trial.passed:
                    success_idx = i
                    break
            assert success_idx != -1

            # find the successful path
            path = []
            while success_idx != -1:
                success_idx = package.parents[success_idx]
                path.append(success_idx)

            for i in range(len(path)-1):
                chosen_idx = path[i]
                rejected_idx = path[i+1]
                pair.append((chosen_idx, rejected_idx))

            for i in range(len(package.trials)):
                if i not in path:
                    pair.append((-1,i))  # -1 indicates root
                
            for chosen_idx, rejected_idx in pair:
                value_pair['chosen'].append(
                    package.generate_prompt(chosen_idx, self.model_type)
                )
                value_pair['reject'].append(
                    package.generate_prompt(rejected_idx, self.model_type)
                )

        return value_pair

    async def sample_with_random(self, packages: List[Package]):
        """
        for a package, sample at most max_trails_per_package trails until getting correct proof
        """
        self.value_model.eval()
        num_trail = 0   
        while num_trail < self.max_trails_per_package:
            futures = []
            for idx, package in enumerate(packages):
                if package.passed:
                    continue

                # randomly select a trail
                trial_indices = [
                        i
                        for i in range(len(package.trials))
                        if not package.trials[i].error and not package.trials[i].passed
                ] + [-1]  # -1 indicates root
                
                selected_idx = random.choice(trial_indices)

                # generate proof with selected trail 
                generate_prompt = package.generate_prompt(selected_idx, self.model_type)
                
                package.parents.append(selected_idx)
                futures.append((idx, generate_prompt,self.req_agent.prove(generate_prompt)))

            for i, generate_prompt, future in futures:
                response_text, success = await future
                packages[i].trials.append(
                    Trial(response=response_text) if success else Trial(error=f"{response_text}")
                )
                self.lean_verifier.verify(packages[i])

            num_trail += 1

    async def sample_with_value(self, packages: List[Package]):
        """
        for a package, sample at most max_trails_per_package trails until getting correct proof
        """
        self.value_model.eval()
        # calculate root scores
        root_value_ids = []
        root_value_masks = []
        for package in packages:
            value_token = self.tokenize_value_prompt(package.generate_prompt(-1, self.model_type))
            root_value_ids.append(value_token["input_ids"])
            root_value_masks.append(value_token["attention_mask"])
        root_value_ids = zero_pad_sequences(root_value_ids, side="left", value=self.tokenizer.pad_token_id)
        root_value_masks = zero_pad_sequences(root_value_masks, side="left")
        with torch.no_grad():
            root_scores = self.value_model(
                input_ids=root_value_ids.squeeze(1).to(torch.cuda.current_device()),
                attention_mask=root_value_masks.squeeze(1).to(torch.cuda.current_device()),
            )
        root_scores = root_scores.cpu().tolist()
        for i, package in enumerate(packages):
            package.root_score = root_scores[i]

        num_trail = 0   
        while num_trail < self.max_trails_per_package:
            futures = []
            for idx, package in enumerate(packages):
                if package.passed:
                    continue

                # value model will guide the sampling tree to select the best trail
                trial_scores = [
                        (i, s)
                        for i, s in enumerate(package.scores)
                        if not package.trials[i].error and not package.trials[i].passed
                ] + [(-1, package.root_score)]  # -1 indicates root

                # caculate revised scores
                # here we view the value-model score as a prior, the revised score is prior / 1 + failures
                # revised_scores = []
                # for idx, score in trial_scores:
                #     failures = package.parents.count(idx)
                #     revised_score = score / (1 + failures)
                #     revised_scores.append((idx, revised_score))

                # revised_scores = trial_scores
                # # select the trail with highest revised score
                # selected_idx, _ = max(revised_scores, key=lambda x: x[1])

                # 1. 分离索引和分数
                candidate_indices = [x[0] for x in trial_scores]
                candidate_scores = torch.tensor([x[1] for x in trial_scores], dtype=torch.float)

                # 2. 计算 Softmax 概率 (可以在这里除以 temperature 来控制随机性，例如 candidate_scores / temp)
                probs = torch.softmax(candidate_scores, dim=0)

                # 3. 根据概率进行采样
                # multinomial 返回的是 candidate_scores 中的下标位置
                sample_pos = torch.multinomial(probs, num_samples=1).item()
                
                # 4. 获取对应的 trial 索引
                selected_idx = candidate_indices[sample_pos]
                
                # generate proof with selected trail 
                generate_prompt = package.generate_prompt(selected_idx, self.model_type)
                
                package.parents.append(selected_idx)
                futures.append((idx, generate_prompt,self.req_agent.prove(generate_prompt)))

            value_package_idx = []
            value_ids = []
            value_masks = []
            for i, generate_prompt, future in futures:
                response_text, success = await future
                packages[i].trials.append(
                    Trial(response=response_text) if success else Trial(error=f"{response_text}")
                )
                verifier_score = self.lean_verifier.verify(packages[i])
                if verifier_score is not None:
                    packages[i].scores.append(verifier_score)
                else:
                    value_token = self.tokenize_value_prompt(generate_prompt)
                    value_package_idx.append(i)
                    value_ids.append(value_token["input_ids"])
                    value_masks.append(value_token["attention_mask"])

            if len(value_ids) > 0:
                value_ids = zero_pad_sequences(value_ids, side="left", value=self.tokenizer.pad_token_id)
                value_masks = zero_pad_sequences(value_masks, side="left")
                with torch.no_grad():
                    scores = self.value_model(
                        input_ids=value_ids.squeeze(1).to(torch.cuda.current_device()),
                        attention_mask=value_masks.squeeze(1).to(torch.cuda.current_device()),
                    )
                scores = scores.cpu().tolist()
                for i, score in enumerate(scores):
                    pkg_idx = value_package_idx[i]
                    packages[pkg_idx].scores.append(score)

            num_trail += 1