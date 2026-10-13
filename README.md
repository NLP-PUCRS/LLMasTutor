# LLM As Tutor

> **Derivative Work Notice & Acknowledgements:**
> This repository is a modified and derived version of the **TEXTOIR** (Text Open Intent Recognition) platform. The base framework for *Unsupervised and Semi-Supervised New Intent Discovery* (USNID) was originally developed by [THUIAR](https://github.com/thuiar/TEXTOIR). We deeply thank the original authors for making their code available under the MIT License.

This repository contains the code and data pipeline for the unsupervised discovery and categorization of actionable intents in academic peer-review feedback, completely bypassing the opacity of generative Large Language Models (LLMs). Our unsupervised pipeline isolates and labels actionable intents without relying on manual annotations, translating complex semantic representations into human-interpretable categories via an automated morphosyntactic analysis.

## ToC

1. [Cite](#cite)
1. [Architecture & Pipeline](#architecture--pipeline)
1. [Datasets](#datasets)
1. [Installation](#installation)
1. [Running Experiments](#running-experiments)
1. [License](#license)

## Cite

If you find this repository or our methodology useful for your research, please cite our paper:
[Paper Link](https://link.springer.com/chapter/10.1007/978-3-032-39898-7_18#citeas)
```bibtex
@InProceedings{10.1007/978-3-032-39898-7_18,
author="Madeira, Caio
and Dopke, Luan
and Losch, Gustavo
and Magnaguagno, Mauricio
and Griebler, Dalvan",
editor="Recamonde-Mendoza, Mariana
and Barros, Rodrigo C.",
title="New Intent Discovery in Peer-Review Critiques: A Morphosyntactic Approach for Interpretable Cluster Labeling",
booktitle="Intelligent Systems",
year="2027",
publisher="Springer Nature Switzerland",
address="Cham",
pages="259--273",
abstract="The academic peer-review process generates a massive volume of unstructured textual feedback that contains highly valuable insights for scientific editorial improvement. While this data holds immense potential for training automated end-to-end revision systems, extracting actionable intents from these reviews remains a profound challenge due to their high semantic diversity and subjective complexity. To bridge this gap, this paper proposes a novel application of New Intent Discovery (NID) techniques for the scientific domain, coupled with an automatic cluster labeling architecture. By evaluating multiple Transformer backbones---SciBERT, BGE, and Specter 2---we extract semantic representations with ARIES and ReAct as input data, and explicitly evaluate the representation degeneration problem through geometric embedding analysis. Finally, to transcend traditional transactional intent discovery, we propose a novel semantic router based on a morphosyntactic analysis and dependency parsing to automatically assign human-interpretable labels to the discovered clusters. Our results demonstrate that this unsupervised approach isolates and labels actionable intents without relying on manual annotations, translating these complex representations into human-interpretable categories, providing a set of foundational suggestion classes and a pipeline that can be incorporated into automated feedback systems for peer-review.",
isbn="978-3-032-39898-7"
}


```

Since this work builds upon the algorithmic foundation of USNID, it is mandatory to cite the original methodology authors:

```bibtex
@inproceedings{zhang-etal-2021-textoir,
    title = "{TEXTOIR}: An Integrated and Visualized Platform for Text Open Intent Recognition",
    author = "Zhang, Hanlei  and Li, Xiaoteng  and Xu, Hua  and Zhang, Panpan and Zhao, Kang  and Gao, Kai",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    pages = "167--174",
    year = "2021",
    url = "[https://aclanthology.org/2021.acl-demo.20](https://aclanthology.org/2021.acl-demo.20)",
    doi = "10.18653/v1/2021.acl-demo.20",
}

@article{DA-ADB, 
    title = {Learning Discriminative Representations and Decision Boundaries for Open Intent Detection},  
    author = {Zhang, Hanlei and Xu, Hua and Zhao, Shaojie and Zhou, Qianrui}, 
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},  
    volume = {31},
    pages = {1611-1623},
    year = {2023}, 
    doi = {10.1109/TASLP.2023.3265203} 
} 

@ARTICLE{USNID,
  author={Zhang, Hanlei and Xu, Hua and Wang, Xin and Long, Fei and Gao, Kai},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Clustering Framework for Unsupervised and Semi-supervised New Intent Discovery}, 
  year={2023},
  doi={10.1109/TKDE.2023.3340732}
}
```

## Architecture & Pipeline

The proposed system is divided into three main modules:

- **Data Preprocessing:** Merges the ARIES and ReAct datasets, removes academic noise, and applies a fine-tuned DistilRoBERTa (82M) model to filter out non-actionable sentences (e.g., general praise, score updates).  

- **Intent Discovery (USNID):** Applies the USNID deep clustering framework. We evaluate the geometric properties of the latent spaces using three Transformer backbones: SciBERT, Specter 2, and BGE.

- **Cluster Labeling (Semantic Router):** A spaCy-based morphosyntactic router that utilizes Part-of-Speech (POS) rules and dependency parsing to extract "action-object" pairs from review comments, generating human-readable labels (e.g., [provide detail], [need iteration] + model method).

## Datasets

We use the following datasets to form our base corpus for intent discovery:

- **ARIES:** Used to extract academic peer-review feedback.
- **ReAct:** Merged with ARIES to provide a robust set of review comments for evaluation and clustering.

## Installation  

We recommend using Anaconda to create the Python environment. Install the required dependencies using the following commands:

```bash
conda create --name llmastutor python=3.13
conda activate llmastutor
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Running Experiments

To run the experiments and execute the pipeline, run the following script:  

```bash
bash TEXTOIR/open_intent_discovery/run.sh
```

## License

This project is licensed under the MIT License.

The core codebase related to TEXTOIR and USNID was authored by THUIAR (Copyright (c) 2022 THUIAR). The modifications, peer-review preprocessing pipeline, and morphosyntactic label router were authored by [Your Name/Institution] (Copyright (c) 2026). See the LICENSE file for more details.
