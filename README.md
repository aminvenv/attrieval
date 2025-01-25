<p style="text-align: center;"> <strong>AttriEval</strong> </p>

AttriEval consists of three modules:
- Data: used to load the data and prompts.
- Generation: used to generate answers for a given question and its top-k retrieved documents as context. 
- Evalution: used to evaluate the attribution quality and attribution bias.

# Installation
Run the following command to install the package:

```bash 
python setup.py install
```
AttriEval is tested with `Python 3.9.0`.

# Load Data 
The first step is to load one of the benchmarks using `load_data_from_disk` function.

```python
from attrieval.data_utils import load_data_from_disk, prepare_and_label_data
att_data = load_data_from_disk('./data/nq/')
```

In the next step, we load proper prompt templates and prepare the data.

Two arguments should be specified for this purpose. `rag_mode` is set with one of the three rag models (i.e., information settings) are used for answer generation. In Vanilla mode (vanilla), we feed documents to the answer generator LLMs without any labels. In informed (base) and counterfactual mode (cf), relevant and non-relevant documents are labeled with two different set of labels. These labels are set according to the `metadata_type` which should be set to `hai` for human versus AI (LLM) authorship. Other types of metadata can be set according to the user preference:

- `hai`: human versus AI (LLM) authorship.
- `gender`: male versus female authorship.
- `race`: black people versus white people authorship.

```python
prep_att_data, sys_prompt_tmplt, user_prompt_tmplt = prepare_and_label_data(att_data, rag_mode='base', metadata_type='hai')
```

# Generation
In order to genereate the answer, three arguments should be specified as follows:

- `model_name`: the LLM identifier.
- `device`: the GPU device id for loading the answer generator model and its corresponding tokenizer.
- `user_token`: if the LLM specified with `model_name` needs authorized access, `hf_token` needs to be set with the user's token.

``` python
from attrieval.generation import ModelHandler
model_handler = ModelHandler(model_name= "meta-llama/Meta-Llama-3-8B-Instruct",
                             device= "auto",
                             user_token= [user/token])
```

```python
generation_att_output = model_handler.generate_with_probs(prep_att_data,
                                                 sys_prompt_tmplt,
                                                 user_prompt_tmplt)
```

# Evaluation 
Given the ground-truth document which contains the answer, and citations provided in an answer, we use precision and recall to assess the quality of attribution for the provided answer of an LLM. The output of AttriEval generation moduel, i.e., `generation_att_output`, includes these required elements. 


```python
from attrieval import evaluation
evaluation.evaluate_prec_recall(generation_att_output)
```

CAB (attribution bias) and CAS (attribution sensitivity) can be computed given the generation outputs for different rag modes. 
For CAB, the output of `vanilla` mode should be compared against the output of the `base` mode.
For CAS, the output of `base` mode should be compared against the output of `cf` mode.

```python
evaluation.cab_evaluation_nq(generation_att_output, generation_att_output_cf)
```

