IGNORE_INDEX = -100

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

prompt_input = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

prompt_no_input = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

import os
import abc
import copy
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any

import torch
import random
import transformers
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from collections import defaultdict

from transformers import DataCollator


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if positions == "all":
        first_n, last_n = last_position // 2, last_position // 2
    elif "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["position"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs


class ReftDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self, task: str, data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", dataset=None, seed=42, max_n_example=None,
        **kwargs,
    ):
        super(ReftDataset, self).__init__()
        result = defaultdict(list)

        # setup
        self.tokenizer = tokenizer
        self.first_n, self.last_n = parse_positions(kwargs["position"]) 
        self.task = task
        self.data_path = data_path
        self.data_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.pad_mode = "first"
        self.fields_to_pad = ["input_ids", "labels"]
        self.fields_to_mask = ["input_ids"]

        # load the dataset
        self.preprocess(kwargs)
        self.task_dataset = self.load_dataset()

        # kwargs settings
        self.postprocess(kwargs)

        # tokenize and intervene
        self.result = []
        for i, data_item in enumerate(tqdm(self.task_dataset)):
            tokenized, last_position = self.tokenize(data_item)
            tokenized = self.compute_intervention_and_subspaces(i, data_item, tokenized, last_position, **kwargs)
            self.result.append(tokenized)

    @abc.abstractmethod
    def tokenize(self, data_item, **kwargs):
        """How to tokenize a single data item. Override this function!"""
        return

    def preprocess(self, kwargs):
        """Preprocessing."""
        return

    def postprocess(self, kwargs):
        """Postprocessing."""
        return
    
    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.result[i])

    def load_dataset(self):
        """Load the dataset (or a portion of it) from HF or a local file."""

        # load the dataset
        if self.dataset is None:
            print("loading data for dataset: ", self.data_path)
            if self.data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=self.data_path)["train"]
            elif self.data_path is not None:
                task_dataset = load_dataset(self.task, self.data_path)[self.data_split]
            else:
                task_dataset = load_dataset(self.task)[self.data_split]
        else:
            task_dataset = self.dataset

        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if self.data_split != "train" else None
        return task_dataset
        
    def get_intervention_locations(self, **kwargs):
        return get_intervention_locations(**kwargs)
    
    def compute_intervention_and_subspaces(self, id: int, data_item, result: dict, last_position: int, **kwargs):
        # compute intervention locs
        intervention_locations = self.get_intervention_locations(last_position=last_position, first_n=self.first_n, 
            last_n=self.last_n, pad_mode=self.pad_mode, **kwargs)
        result["intervention_locations"] = intervention_locations
        result["id"] = id
            
        # add a single padding token BEFORE input_ids and fix everything
        if self.pad_mode == "first":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((torch.tensor([IGNORE_INDEX,]), result[field]))
                else:
                    result[field] = torch.cat((torch.tensor([self.tokenizer.pad_token_id,]), result[field]))
            result["intervention_locations"] = (torch.IntTensor(result["intervention_locations"]) + 1).tolist()
        elif self.pad_mode == "last":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels" and field in result:
                    result[field] = torch.cat((result[field], torch.tensor([IGNORE_INDEX,])))
                else:
                    result[field] = torch.cat((result[field], torch.tensor([self.tokenizer.pad_token_id,])))
        
        # attention masks
        if len(self.fields_to_mask) == 1:
            result["attention_mask"] = (result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id).int()
        else:
            for field in self.fields_to_mask:
                result[f"{field}_mask"] = (result[field] != self.tokenizer.pad_token_id).int()

        # subspaces
        if "subspaces" in data_item:
            num_interventions = kwargs["num_interventions"]
            share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
            if share_weights:
                num_interventions = num_interventions // 2
            # we now assume each task has a constant subspaces
            _subspaces = [data_item["subspaces"]] * num_interventions
            result["subspaces"].append(_subspaces)

        return result


class ReftRawDataset(Dataset):

    def __init__(
        self, task: str, data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", dataset=None, seed=42, max_n_example=None, 
        **kwargs,
    ):
        super(ReftRawDataset, self).__init__()
        result = defaultdict(list)

        if dataset is None:
            print("loading data for dataset: ", data_path)
            if data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=data_path)[data_split]
            else:
                task_dataset = load_dataset(data_path)[data_split]
        else:
            task_dataset = dataset
        if max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=seed)
            task_dataset = task_dataset.select(range(max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if data_split != "train" else None
        first_n, last_n = parse_positions(kwargs["position"])
        
        # tokenize and intervene
        for i, data_item in enumerate(tqdm(task_dataset)):
            base_prompt = data_item["instruction"]
            base_input = base_prompt + data_item["output"] + tokenizer.eos_token

            # tokenize
            base_prompt_ids = tokenizer(
                base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)
            if data_split == "train":
                base_input_ids = tokenizer(
                    base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                output_ids = copy.deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                    
                result["input_ids"].append(base_input_ids)
                result["labels"].append(output_ids)
            else:
                # print("Assuming test split for now")
                result["input_ids"].append(base_prompt_ids)
            last_position = base_prompt_length
                
            # get intervention locations
            intervention_locations = self.get_intervention_locations(
                last_position=last_position, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="first",
                **kwargs
            )
            result["intervention_locations"].append(intervention_locations)
            result["id"].append(i)
            
            # add a single padding token BEFORE input_ids and fix everything
            result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
            if data_split == "train":
                result["labels"][-1] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"][-1]))
            result["intervention_locations"][-1] = (torch.IntTensor(result["intervention_locations"][-1]) + 1).tolist()
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())
            if "subspaces" in data_item:
                num_interventions = kwargs["num_interventions"]
                share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
                if share_weights:
                    num_interventions = num_interventions // 2
                # we now assume each task has a constant subspaces
                _subspaces = [data_item["subspaces"]] * num_interventions
                result["subspaces"].append(_subspaces)
        
        self.input_ids = result["input_ids"]
        self.attention_mask = result["attention_mask"]
        self.intervention_locations = result["intervention_locations"]
        self.labels = result["labels"] if "labels" in result else None
        self.subspaces = result["subspaces"] if "subspaces" in result else None
        self.id = result["id"]

    def get_intervention_locations(self, **kwargs):
        return get_intervention_locations(**kwargs)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return_dict = dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            intervention_locations=self.intervention_locations[i],
            id=self.id[i],
        )
        if self.labels is not None:
            return_dict["labels"] = self.labels[i]
        if self.subspaces is not None:
            return_dict["subspaces"] = self.subspaces[i]
        return return_dict



class ReftClassificationDataset(ReftDataset):
    """
    A ReftClassificationDataset only contains a single text field
    that we tokenize, intervene on a prefix + suffix of, and
    compute subspace settings for. This is intended for classification
    tasks.

    Remember to pass in the input_field and label_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.input_field = kwargs["input_field"]
        self.label_field = kwargs["label_field"]

    def tokenize(self, data_item):
        result = {}
        
        # input
        input_ids = self.tokenizer(data_item[self.input_field], max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(input_ids)
        last_position = base_prompt_length - 1
        result["input_ids"] = input_ids

        # labels
        if self.label_field == self.input_field:
            result["labels"] = input_ids.clone()
        elif self.label_field is not None:
            labels = self.tokenizer(data_item[self.label_field], max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")["input_ids"][0]
            result["labels"] = labels
            
        return result, last_position


class ReftGenerationDataset(ReftDataset):
    """
    A ReftGenerationDataset contains an instruction and a 
    completion for each data item. We intervene on a prefix + suffix
    of *only the instruction*. This is suitable for generation tasks
    where you don't want inference overhead during decoding.

    Remember to pass in the prompt_field and completion_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.prompt_field = kwargs["prompt_field"]
        self.completion_field = kwargs["completion_field"]

    def tokenize(self, data_item):
        result = {}
        
        # prompt
        prompt_ids = self.tokenizer(data_item[self.prompt_field], max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1
        
        # input
        full_input = data_item[self.prompt_field] + data_item[self.completion_field] + self.tokenizer.eos_token
        input_ids = self.tokenizer(full_input, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        result["input_ids"] = input_ids

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids
            
        return result, last_position


class ReftSupervisedDataset(ReftDataset):
    """
    Alpaca-style supervised dataset. We intervene on a prefix + suffix
    of the input. This is suitable for supervised fine-tuning tasks.

    Remember to pass in the input_field, output_field, and instruction_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.input_field = kwargs["input_field"]
        self.output_field = kwargs["output_field"]
        self.instruction_field = kwargs["instruction_field"]

    def tokenize(self, data_item):
        result = {}

        # prompt
        if self.input_field not in data_item or data_item[self.input_field] == "":
            base_prompt = prompt_no_input % (data_item[self.instruction_field])
        else:
            base_prompt = prompt_input % (data_item[self.instruction_field], data_item[self.input_field])
        prompt_ids = self.tokenizer(base_prompt, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1
        
        # input
        base_input = base_prompt + data_item[self.output_field] + self.tokenizer.eos_token
        input_ids = self.tokenizer(base_input, max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
        result["input_ids"] = input_ids

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids
            
        return result, last_position


def make_last_position_supervised_chat_data_module(tokenizer: transformers.PreTrainedTokenizer, model, inputs, outputs, nonstop=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]
    
        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token
    
        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        
        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]])
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_last_position_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model, inputs, outputs, nonstop=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]
    
        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token
    
        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        
        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]])
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class ReftPreferenceDataset(ReftDataset):
    """
    Different from ReftSupervisedDataset where we have
    (x, y)
    ReftPreferenceDataset contains (x, y1, y2) where y1 and y2
    are constrastive pairs.
    ReFT training objective is to generate y2, given (x, y1) and
    the intervention.
    """

    def preprocess(self, kwargs):
        self.input_field = kwargs["input_field"]
        self.instruction_field = kwargs["instruction_field"]
        self.chosen_output_field = kwargs["chosen_output_field"]
        self.rejected_output_field = kwargs["rejected_output_field"]

    def tokenize(self, data_item):
        result = {}

        if self.input_field not in data_item or data_item[self.input_field] == "":
            base_prompt = prompt_no_input % (data_item[self.instruction_field])
        else:
            base_prompt = prompt_input % (data_item[self.instruction_field], data_item[self.input_field])
        # base input takes rejected output to steer away from.
        base_input = base_prompt + data_item[self.rejected_output_field] + self.tokenizer.eos_token

        # tokenize
        base_prompt_ids = self.tokenizer(
            base_prompt, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.data_split == "train":
            base_input_ids = self.tokenizer(
                base_input, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            # base output takes chosen output to steer towards to.
            base_output = base_prompt + data_item[self.chosen_output_field] + self.tokenizer.eos_token

            base_output_ids = self.tokenizer(
                base_output, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            output_ids = base_output_ids
            output_ids[:base_prompt_length] = IGNORE_INDEX

            # padding! needs to be cautious here. let's unpack:
            # pad inputs with pad_token_id so that attention masks can ignore these tokens.
            # pad outputs with IGNORE_INDEX so that loss calculation can ignore these tokens.
            # and the goal is to have input and output have the same length.
            max_length = max(base_input_ids.size(0), output_ids.size(0))
            input_pad_length = max_length - base_input_ids.size(0)
            output_pad_length = max_length - output_ids.size(0)

            input_pad_tensor = torch.full((input_pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            output_pad_tensor = torch.full((output_pad_length,), IGNORE_INDEX, dtype=torch.long)

            base_input_ids_padded = torch.cat((base_input_ids, input_pad_tensor), dim=0)
            output_ids_padded = torch.cat((output_ids, output_pad_tensor), dim=0)

            result["input_ids"] = base_input_ids_padded
            result["labels"] = output_ids_padded
        else:
            # print("Assuming test split for now")
            result["input_ids"] = base_prompt_ids

        last_position = base_prompt_length
        return result, last_position


class ReftRewardDataset(ReftDataset):

    def preprocess(self, kwargs):
        self.conv_A_field = kwargs["conv_A_field"]
        self.conv_B_field = kwargs["conv_B_field"]
        self.conv_A_reward_field = kwargs["conv_A_reward_field"]
        self.conv_B_reward_field = kwargs["conv_B_reward_field"]
        self.fields_to_pad = ["chosen_output", "rejected_output"] # pad both chosen and rejected with dummy tok
        self.fields_to_mask = ["chosen_output", "rejected_output"] # -> chosen_output_mask, rejected_output_mask

    def tokenize(self, data_item):
        result = {}

        # generate prompt format
        chosen_output = self.tokenizer.apply_chat_template(
            data_item[self.conv_A_field], tokenize=False, add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
        rejected_output = self.tokenizer.apply_chat_template(
            data_item[self.conv_B_field], tokenize=False, add_generation_prompt=False).replace(self.tokenizer.bos_token, "")
        
        # reward
        result["chosen_reward"] = data_item[self.conv_A_reward_field]
        result["rejected_reward"] = data_item[self.conv_B_reward_field]

        # swap so that chosen is better
        if result["chosen_reward"] < result["rejected_reward"]:
            chosen_output, rejected_output = rejected_output, chosen_output
            result["chosen_reward"], result["rejected_reward"] = result["rejected_reward"], result["chosen_reward"]

        # tokenize
        chosen_ids = self.tokenizer(
            chosen_output, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        rejected_ids = self.tokenizer(
            rejected_output, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = 0
        for i in range(min(len(chosen_ids), len(rejected_ids))):
            base_prompt_length += 1
            if chosen_ids[i] != rejected_ids[i]:
                break
        last_position = base_prompt_length - 1

        result["chosen_output"] = chosen_ids
        result["rejected_output"] = rejected_ids
        return result, last_position
    
@dataclass
class InterventionDataCollator(object):
    """Collate examples for Intervention."""
    
    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_intervention_len = max([len(inst["intervention_locations"][0]) for inst in instances])
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])
        
        for inst in instances:
            non_pad_len = len(inst["input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [[len(inst["input_ids"]) for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))]])
            _intervention_mask_paddings = torch.tensor(
                [0 for _ in range(max_intervention_len - len(inst["intervention_locations"][0]))])
            inst["intervention_locations"] = torch.cat([inst["intervention_locations"], _intervention_location_paddings], dim=-1).int()
            inst["intervention_masks"] = torch.cat([_intervention_mask, _intervention_mask_paddings], dim=-1).int()
            inst["prompt_intervention_masks"] = inst["intervention_masks"].clone()
            inst["prompt_intervention_masks"][inst["prompt_lengths"]:] = 0 # mask out the intervention locations after prompt length

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)])
            inst["input_ids"] = torch.cat((inst["input_ids"], torch.tensor([self.tokenizer.pad_token_id]), _input_id_paddings)).int()

            _label_paddings = torch.tensor([-100 for _ in range(max_seq_len - non_pad_len+1)])
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))
            
            inst["attention_mask"] = (inst["input_ids"] != self.tokenizer.pad_token_id).int()

        batch_inputs = self.data_collator(instances)
        return batch_inputs

def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer, df, 
    dataset_category="continuation",
    positions="all", # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    **kwargs
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0
    
    all_base_input_ids, all_intervention_locations, all_output_ids,  = [], [], []
    all_prompt_lengths = []
    for _, row in df.iterrows():
        _input, _output = row["input"], row["output"]
        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        if positions is None or positions == "all_prompt":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_prompt_length)]])
        elif positions == "all":
            intervention_locations = torch.tensor([[i for i in range(prefix_length, base_length)]])
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length - prefix_length, 
                first_n=first_n, 
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by prefix length
            shifted_intervention_locations = [[loc + prefix_length for loc in intervention_locations[0]]]
            intervention_locations = shifted_intervention_locations
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_prompt_lengths.append(torch.tensor(base_prompt_length - 1)) # exclude bos token
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "prompt_lengths": all_prompt_lengths,
    })
    train_dataset.set_format(
        type='torch', columns=[
            'input_ids', 'intervention_locations', 'prompt_lengths', 'labels'])

    data_collator_fn = transformers.DefaultDataCollator(
        return_tensors="pt"
    )
    data_collator = InterventionDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


