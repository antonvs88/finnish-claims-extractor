---
license: apache-2.0
datasets:
- airtrain-ai/fineweb-edu-fortified
- bigcode/starcoderdata
- HuggingFaceTB/smollm-corpus
- allenai/peS2o
- uonlp/CulturaX
- HPLT/HPLT2.0_cleaned
- cis-lmu/Glot500
- ltg/saami-web
- rpa020/SALT-train
language:
- fi
- sv
- en
- se
pipeline_tag: fill-mask
library_name: transformers
---
<img src="images/finnish_modernbert.png" alt="Finnish ModernBERT" width="600" height="600">

# Finnish ModernBERT Model Card
Finnish ModernBERT base is an encoder model following the ModernBERT architecture, pretrained on Finnish, Swedish, English, Code, Latin, and Northern Sámi. 
It was trained on 410.2B tokens. **This model can theoretically extrapolate to a context length of 128K.**
Training was conducted on the [LUMI supercomputer](https://www.lumi-supercomputer.eu/).
The project aimed to train multilingual encoder models that support long context and all official Finnish languages¹.

¹Multiple Sámi languages are spoken in Finland, but Northern Sámi is the most widespread and thus included in the training data. English is not the official language of Finland, but it is widely used. Latin was included for potential clinical use.

Full descriptions of training, data and evaluation are available in the [article](https://arxiv.org/abs/2511.09213). 

## Table of Contents
1. [Model Overview](#model-overview)
2. [Evaluation](#evaluation)
3. [Data](#data)
4. [Training](#training)
3. [Ethical Considerations](#ethical-considerations)
4. [Aknowledgements](#aknowledgements)
5. [Licence](#licence)
6. [Citation information](#citation-information)

## Model Overview
| Hyperparameter | Value  |
| :------------- | :----: |
| n_parameters | 143M |
| n_layers | 22 |
| RoPE base | 10K / 1M |
| vocab_size | 42K |
| sequence_length | 16K / 128K |

## Evaluation
The Finnish ModernBERT models were competitive with other multilingual models ([XLM-R-large](https://huggingface.co/FacebookAI/xlm-roberta-large) and [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)) 
on short-context NLU tasks in Finnish, Swedish, and English, where
XLM-R-large was the strongest model. The Finnish ModernBERTs are the strongest multilingual encoder models in out-of-domain retrieval
tasks, outperforming the others with a large margin.

## Data
We used text datasets from diverse sources, including web crawls, news, scientific articles, classical literature, historical texts, Wikipedia, forums, and authoritative sources. 
Sources underwent various levels of pre-processing, including the removal of low-quality text or boilerplate, PII removal, and deduplication.
Note that more datasets were used for the training than listed in this repository metadata.

## Training
Pretraining was done using Distributed Data Parallelism, AdamW with ZeroRedundancyOptimizer, and the WSD learning rate schedule.
We describe the training as a three-step process where:
1. Models' parameters are first optimized for short context representations of the tokens (stable phase)
2. Refine the token representations for longer dependencies (context extension phase)
3. Reinforce the representations for the inputs that we believe the model will be used for in the future (annealing phase)

## Ethical Considerations and Limitations
Finnish ModernBERTs' training data include sources regarded as biased and harmful, and the models' outputs may mirror these biased properties. 
The training data were not filtered for toxic, harmful, or offensive content to serve various use cases.
The representations produced by the models should not be used without
caution and without evaluating their effects on vulnerable population groups.

## Licence
Finnish ModernBERT base is released under the Apache 2.0 license.

## Aknowledgements
We acknowledge CSC, IT Center for Science, Finland, for awarding this project access to the LUMI supercomputer, 
owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium. We acknowledge the HPLT-project for supporting this research. 
This project has received funding from 
the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070350, 
and it has also received funding from the Finnish Cultural Foundation.
We thank [The Language Bank of Finland](https://www.kielipankki.fi/) for additional resources for Finnish, Finland-Swedish, and Swedish.

## Citation information
If you use Finnish ModernBERTs or need to reference the work, please use the citation below: 
~~~
@misc{reunamo2025pretrainingfinnishmodernberts,
      title={Pretraining Finnish ModernBERTs}, 
      author={Akseli Reunamo and Laura-Maria Peltonen and Hans Moen and Sampo Pyysalo},
      year={2025},
      eprint={2511.09213},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.09213}, 
}
~~~
