# LAMBrrr

This package provides an efficient implementation of the LAMB optimizer in PyTorch 2. The optimizer is horizontally fused using `torch._foreach` ops and vertically fused using `torch.compile` and Triton. This improves performance compared with existing implementations, especially for models with a large number of parameter tensors. 10% faster than existing PyTorch implementations, but still slower than Nvidia apex's FusedLAMB [4].

The optimizer was tested using a Turing Nvidia GPU, but should run on all devices supported by [Triton](https://triton-lang.org/).

Note: the formulation of the LAMB optimizer changed throughout versions of the original paper. This package implements the latest formulation at the time of writing (v5).

## Why LAMB?

LAMB enables models to be trained using larger batch sizes than ADAM. This is useful for large models to better exploit 3d/pipeline parallelism or for contrastive models - among other purposes.

## Install

Install using `pip`:

    pip install https://github.com/nopperl/pytorch-fused-lamb

The package requires `torch>=2.2.0` and `triton>=2.21`, which need to be installed seperately, e.g. using `pip`:

    pip install torch triton

## Usage

Import the optimizer class:

    from pytorch_fused_lamb import Lamb

The `Lamb` optimizer can be used like any other `torch.optim.Optimizer`. For a PyTorch model:

```diff
-from torch.optim import Adam
+from pytorch_fused_lamb import Lamb
from torch import nn

dataset = ...
model = ...
-optimizer = Adam(model.parameters(), lr=1e-3)
+optimizer = Lamb(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
for (x, y) in dataset:
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
```

### HuggingFace

```diff
-from torch.optim import Adam
+from pytorch_fused_lamb import Lamb
from torch import nn
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

train_dataset = ...
eval_dataset = ...

-optimizer = Adam(model.parameters(), lr=1e-3)
+optimizer = Lamb(model.parameters(), lr=1e-3)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None)
)

# Start training
trainer.train()
```

## Benchmark

Tested with `torch==2.4.0a0+git1346ebf`, `triton==3.0.0` and CUDA 11.8.

Comparison with a reference implementation [3] and the nvidia/apex FusedLAMB [4] optimizer (only supports CUDA) is in [benchmark.ipynb](benchmark.ipynb).

Results:

```
[------------  -----------]
                 |  runtime
1 threads: ----------------
      reference  |    26.3 
      apex       |    10.4 
      compiled   |    23.6 

Times are in milliseconds (ms).
```


## TODO
  * selectively exclude layers from lr adaptation (`exclude_from_layer_adaptation`)
  * weight norm clipping
  * grad norm clipping
  * trust ratio clipping (LAMBC)
  * compare convergence with Adam and LAMB with 4x batch size (requires more GPU resources)
  * AdamW update
  * Adagrad update


## References
  1. Original paper (version 5): https://arxiv.org/abs/1904.00962v5
  2. Official TensorFlow implementation: https://github.com/tensorflow/addons/blob/3380b3ccf906f20dbef49c05906e6b6dabf479cf/tensorflow_addons/optimizers/lamb.py
  3. PyTorch implementation used as reference: https://github.com/huggingface/pytorch-image-models/blob/59b3d86c1d69fe85ccb5bbdb2f1711eadae6e4a7/timm/optim/lamb.py
  4. fused implementation of LAMB in CUDA: https://github.com/NVIDIA/apex/810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c/apex/optimizers/fused_lamb.py
  5. Official PyTorch implementation of Adam: https://github.com/pytorch/pytorch/blob/1ea2f1eaa19176b8b5b778bb317203dc7f4fd3dc/torch/optim/adam.py
  6. Benchmarking vertical fusion of torch optimizers: https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669

### Other implementations
  * https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py 
  * https://github.com/skyday123/pytorch-lamb
  * https://github.com/Leiwx52/Pytorch-LAMB
  * https://github.com/kozistr/pytorch_optimizer (implements v3, i.e. no debiasing)
  * https://github.com/cybertronai/pytorch-lamb (computed values do not correspond with official implementation, see https://github.com/cybertronai/pytorch-lamb/issues/10), used in:
    * https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/lamb.html
    * https://lightning-flash.readthedocs.io/en/0.5.0/_modules/flash/core/optimizers/lamb.html
