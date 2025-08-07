from typing import List
from abc import ABC,abstractmethod
import ray
import torch
import asyncio
from openrlhf.trainer.ppo_utils.experience_maker import Experience, SamplesGenerator


class SamplesGeneratorAsync(SamplesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        # 使用 asyncio.run 来运行异步版本
        return asyncio.run(self._generate_vllm_async(all_prompts, all_labels, **kwargs))

    async def _generate_vllm_async(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)
        tokenizer = self.tokenizer
        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)

        #######################
        class SamplingTree(ABC):
            def __init__(self):
                self.prompt_to_responses = {}
                self.max_rounds = args.max_rounds
                self.sampling_params = SamplingParams(
                    temperature=kwargs.get("temperature", 1.0),
                    top_p=kwargs.get("top_p", 1.0),
                    top_k=kwargs.get("top_k", -1),
                    max_tokens=kwargs.get("max_new_tokens", 1024),
                    min_tokens=kwargs.get("min_new_tokens", 1),
                    skip_special_tokens=kwargs.get("skip_special_tokens", False),
                )

            @abstractmethod
            async def generate_async(self, prompt, **kwargs):
                pass

            def get_prompt_to_responses(self):
                return self.prompt_to_responses
            
        class SamplingTreeV1(SamplingTree):
            async def generate_async(
                self,
                prompt,
                depth=0,
                **kwargs,
            ):
                assert len(llms) == 1, "Async generation is only supported for a single LLM"
                if depth >= self.max_rounds:
                    return 0

                async def process_single_sample():
                    # Generate responses for the prompt
                    ref = llms[0].add_request.remote(
                        sampling_params=self.sampling_params,
                        prompts=[prompt],
                        labels=[None],  # Labels are not used in this case
                        max_length=truncate_length,
                        hf_tokenizer=tokenizer,
                    )
                    output = await asyncio.to_thread(ray.get, ref)
                    if output["pass"] == False:
                        assert output["reward"] == 0, "Reward should be 0 for non-passing outputs"
                        # expand the node
                        revision_reward = await self.generate_async(
                            prompt=output["next_prompt"],
                            depth=depth + 1,
                        )
                        output["reward"] = revision_reward
                    return output

                # Create concurrent tasks for all samples
                tasks = [process_single_sample() for _ in range(n_samples_per_prompt)]
                outputs = await asyncio.gather(*tasks)

                # Store all outputs and calculate reward sum
                reward_sum = 0
                for output in outputs:
                    self.prompt_to_responses.setdefault(prompt, []).append(output)
                    reward_sum += output["reward"]

                return reward_sum / n_samples_per_prompt

        class SamplingTreeV2(SamplingTree):
            def __init__(self):
                super.__init__()
                self.failure_queue = []

            async def generate_async(
                self,
                prompt,
                round=0,
                **kwargs,
            ):
                """
                    For this version, 
                        prompt_to_responses: store all the passing outputs and the failing outputs that have been expanded
                        failure_queue: store the failing outputs that have not been expanded yet
                """
                if round >= self.max_rounds:
                    return 0
                # TODO(tr) maybe moving the llm parallel to different original prompts is better 
                refs = []
                batch_size = (n_samples_per_prompt + len(llms) - 1) // len(llms)
                for i, llm in enumerate(llms):
                    for _ in range(i * batch_size, min((i + 1) * batch_size, n_samples_per_prompt)):
                        refs.append(
                            llm.add_request.remote(
                                sampling_params=self.sampling_params,
                                prompts=[prompt],
                                labels=[None],
                                max_length=truncate_length,
                                hf_tokenizer=tokenizer,
                            )
                        )
                outputs = await asyncio.to_thread(ray.get, refs)

                for output in outputs:
                    if output["pass"] == False:
                        # store the output for future revision
                        self.failure_queue.append((output["heuristic"], output))

                if len(self.failure_queue) > 0:
                    # get the best output
                    self.failure_queue.sort(key=lambda x: x[0], reverse=True)
                    best_potential = self.failure_queue.pop(0)[1]
                    revision_reward = await self.generate_async(
                        prompt=best_potential["next_prompt"],
                        round=round + 1,
                    )
                    best_potential["reward"] = revision_reward
                
                for output in outputs:
                    self.prompt_to_responses.setdefault(prompt, []).append(output)

                return sum(output["reward"] for output in outputs) / n_samples_per_prompt


        if args.sampling_tree == "v1":
            sampling_tree = SamplingTreeV1()
        elif args.sampling_tree == "v2":
            sampling_tree = SamplingTreeV2()
        else:
            raise ValueError(f"Unknown sampling tree version: {args.sampling_tree}")
        
        tasks = [sampling_tree.generate_async(prompt) for prompt in all_prompts]
        await asyncio.gather(*tasks)
        prompt_groups = sampling_tree.get_prompt_to_responses()
        
        

        eos_token_id = self.tokenizer.eos_token_id
        # Reorder outputs to keep same prompts together
        # This is very important for REINFORCE++-baseline/GRPO/RLOO
        all_outputs = []
        for prompt in prompt_groups.keys():
            all_outputs.extend(prompt_groups[prompt])

        # Process outputs one by one
        experiences_list = []
        for output in all_outputs:
            # Tokenize observation
            observation_tokens = self.tokenizer(output["observation"], add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            tokenized_observation = observation_tokens.tolist()
            if observation_tokens[-1] != eos_token_id:
                tokenized_observation.append(eos_token_id)

            # Convert action ranges to token indices
            tokenized_ranges = []
            for start, end in output["action_ranges"]:
                # Get token indices for the entire observation up to end
                full_tokens = self.tokenizer(
                    output["observation"][:end], add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
                # Get token indices for the entire observation up to start
                start_tokens = self.tokenizer(
                    output["observation"][:start], add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
                # Calculate token indices
                tokenized_ranges.append((len(start_tokens), len(full_tokens)))
            if observation_tokens[-1] != eos_token_id:
                tokenized_ranges[-1] = (tokenized_ranges[-1][0], tokenized_ranges[-1][1] + 1)

            # Create tensors
            sequences = torch.tensor(tokenized_observation)
            attention_mask = torch.tensor([1] * len(tokenized_observation))

            # Create action mask based on tokenized action_ranges
            action_mask = torch.zeros_like(attention_mask)
            # Mark action positions in the mask
            for start, end in tokenized_ranges:
                action_mask[start:end] = 1

            # Apply length limit
            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")

            # Calculate response length (distance between first and last 1)
            ones_indices = torch.where(action_mask)[0]
            response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
            total_length = attention_mask.float().sum()
            is_clipped = total_length >= truncate_length

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
                "reward": torch.tensor([output["reward"]]),
                "score": torch.tensor([output["scores"]]),
            }

            # Process extra_logs
            extra_logs = output.get("extra_logs", {})
            for key, value in extra_logs.items():
                info[key] = torch.tensor([value.item()])

            experience = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                prompts=[output["prompt"]],
                labels=[output["label"]],
                rewards=torch.tensor([output["reward"]]),
                scores=torch.tensor([output["scores"]]),
                info=info,
            )
            # batch*n_samples_per_batch
            experiences_list.append(experience)

        return experiences_list
