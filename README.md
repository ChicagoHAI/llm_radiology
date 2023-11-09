<!-- # Using DeepSpeed for faster inference

We first need to import packages:

```python
import transformers
from transformers.models.t5.modeling_t5 import T5Block
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import deepspeed
import torch
```

DeepSpeed can be run with models or pipelines. Since our report-cleaning task is fairly simple, I opted to use pipeline, which can be set up using the following code:

```python
pipe = pipeline("text2text-generation", model="google/flan-t5-XXL", device='cuda:0')
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=1,
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')},
)
```

The function init_reference() wraps the model inside an InferenceEngine object so that inference can be optimized. mp_size refers to model-parallel size, which indicates the number of GPUs the model can be split into. In our case, Flan-T5-XXL fits inside 1 GPU, so mp_size is 1.

injection_policy indicates which layers of the transformer we want to optimize. I believe for [models](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py) that are supported by DeepSpeed, to optimize, we only need to set replace_with_kernek_inject=True, but for models not supported, we need to specify the layers as above.

Then, doing inference is simply

```python
outputs = pipe(inputs)
```

To run a python script with DeepSpeed, the command is

```
deepspeed --num_gpus n python_script.py
``` -->

# Codebase for Pragmatic Radiology Report Generation

We expose the following functionalities discussed in the paper
* Report cleaning with large language models
* Training an image classifier for detecting positive findings
* Finetuning LLaMA on predicted conditions and indications
* Generating reports with Pragmatic LLaMA
* Evaluating generated reports

This code mainly works for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), but can be adapted slightly to work with other chest X-ray datasets. We will discuss the relevant changes in each section. Make sure to obtain the license if you are working with MIMIC-CXR.

TODO: update requirements.txt

## Obtaining the relevant data

The input to our model is a tuple of (image, indication). Please follow the relevant dataset instructions to obtain them, especially the indication section. For MIMIC-CXR, you can use [create_section_files.py](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt) to extract the indication, findings, and impression sections.

## Report cleaning with large language models

```
deepspeed --num_gpus=<num_gpus> report_cleaning.py --chexbert_path <path_to_chexbert> --dataset_path <path_to_reports> --output_dir <path_to_output_directory>
```

This cleaning method works best for one sentence at a time. The process to split sentences and keep track of their report IDs can be specific to the dataset, so we leave that implementation to the user.



