from random import seed as random_seed
from typing import Optional, Type
from unittest import TestCase, main, skip

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaTokenizerFast, set_seed
import torch
from torch import nn

from pytorch_fused_lamb import Lamb
from .reference import Lamb as ReferenceLamb


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataset(dataset_id, cache_dir=None, max_sequence_length=None) -> Dataset:
    dataset = load_dataset(dataset_id, name="generation", split="validation", cache_dir=cache_dir)
    dataset = dataset.with_format("pt")
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    if max_sequence_length is not None and max_sequence_length > 0:
        tokenizer.model_max_length = max_sequence_length
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def encode(example):
        return tokenizer(example["question"] + " " + example["best_answer"], padding="max_length", truncation=True, return_tensors="pt")
    tokenized_dataset = dataset.map(encode)
    return tokenized_dataset


def run_optimizer(optimizer: Type[torch.optim.Optimizer], optimizer_hyperparams: Optional[dict] = None, batch_size=32, seed=123):
    """ Initializes a toy model and optimizes the parameters using a randomly-generated dataset. """
    random_seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = "cpu"
    model = ToyNet()
    data = torch.randn(batch_size, 10, device=device)
    target = torch.randn(batch_size, 1, device=device)

    common_hyperparams = {
        "lr": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.,
    }
    if optimizer_hyperparams:
        common_hyperparams.update(optimizer_hyperparams)
    optimizer = optimizer(model.parameters(), **common_hyperparams)
    criterion = nn.MSELoss()
    for _ in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model.parameters()


def run_optimizer_large(optimizer_type: Type[torch.optim.Optimizer], tokenized_dataset, optimizer_hyperparams: Optional[dict] = None, max_sequence_length=64, device="cpu", seed=123, deterministic=True, compile_optimizer=False):
    """ Initializes a LLaMA model and optimizes the parameters using the provided dataset. """
#    set_seed(seed, deterministic=deterministic)
    # Replace below two lines with above line for transformers>4.39.2
    set_seed(seed)
    torch.use_deterministic_algorithms(deterministic)

    # Very tiny LLaMA to use little memory
    model_config = LlamaConfig(max_position_embeddings=max_sequence_length, vocab_size=32000, num_attention_heads=4, num_key_value_heads=4, hidden_size=16, intermediate_size=64)
    model = AutoModelForCausalLM.from_config(model_config).to(device)

    common_hyperparams = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.,
    }
    if optimizer_hyperparams:
        common_hyperparams.update(optimizer_hyperparams)
    optimizer = optimizer_type(model.parameters(), **common_hyperparams)
    if compile_optimizer:
        optimizer.step = torch.compile(optimizer.step)
    criterion = nn.CrossEntropyLoss()
    for example in tokenized_dataset:
        optimizer.zero_grad()
        input_ids = example["input_ids"].to(device)
        attention_mask = example["attention_mask"].to(device)
        labels = input_ids[..., 1:].contiguous()
        outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.clone())
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
    return model.parameters()


class ComparisonTest(TestCase):
    """ Tests that the LAMB optimizer converges similarly to reference implementations. """
    dataset_id = "truthful_qa"
    cache_dir = "/tmp/test-cache"
    seed = 123

    def test_compare_with_reference(self):
        """ Compare the LAMB optimizer with a reference implementation on a toy model. """
        parameters = run_optimizer(Lamb, seed=self.seed)
        reference_parameters = run_optimizer(ReferenceLamb, {"grad_averaging": False, "always_adapt": True}, seed=self.seed)
        for p, ref_p in zip(parameters, reference_parameters):
            torch.testing.assert_close(p, ref_p)

    def test_compare_with_reference_on_large_model(self):
        """ Compare the LAMB optimizer with a reference implementation on a LLaMA model. """
        max_sequence_length = 64
        # Small test dataset
        tokenized_dataset = create_dataset(self.dataset_id, cache_dir=self.cache_dir, max_sequence_length=max_sequence_length)
        tokenized_dataset = tokenized_dataset.select(range(5))
        
        parameters = run_optimizer_large(Lamb, tokenized_dataset, seed=self.seed)
        reference_parameters = run_optimizer_large(ReferenceLamb, tokenized_dataset, {"grad_averaging": False, "always_adapt": True}, seed=self.seed)

        for p, ref_p in zip(parameters, reference_parameters):
            # lenient comparison as resulting params can unfortunately be slightly off
            torch.testing.assert_close(p, ref_p, atol=1e-3, rtol=1e-1)
        

if __name__ == '__main__':
    main()

