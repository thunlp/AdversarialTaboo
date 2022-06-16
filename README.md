# Adversarial Taboo

Code and data for the paper AAAI 2021 paper [Adversarial Language Games for Advanced Natural Language Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/17676).

# Introduction
This paper introduces Adversarial Taboo, an adversarial language game in which multiple agents with conflicting goals compete with each other via natural language interactions. This repo provides the code and data for the experiments.

# Requirements and Installation
+ python==3.8
+ nltk==3.7 
+ torch>=1.6.0
+ transformers==4.17.0

# Data Preparation
The game in the paper is built on existing OpenQA and conversation models. In our experiments, we focus on studying the attack and defense strategies. We generate the question-answer pairs and post-response pairs in advance, and sample the pairs according to the strategies to simulate a game, which is more efficient and convenient.

Specifically, to generate the pair data, we use the following repos and raw data:
### OpenQA
+ Download raw data from [Wikipedia dump](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia)
+ Generate questions from [unsupervised question generation](https://github.com/facebookresearch/UnsupervisedQA)
+ Given a question, retrieve the related documents using [BM25 document retrieval](https://github.com/elastic/elasticsearch)
+ Answer the questions using [BERT](https://github.com/huggingface/transformers) or [DocQA](https://github.com/allenai/document-qa) model

### Chatbot
+ Obtain raw data of Reddit from [here](https://github.com/thunlp/ConceptFlow)
+ Given the post, generate responses using [DialoGPT](https://github.com/microsoft/DialoGPT) or [ConceptFlow](https://github.com/thunlp/ConceptFlow) model
+ Train the post classifier (for neural-based and API-based attack) using [BERT](https://github.com/huggingface/transformers) model

We provide the resultant ready-to-use data, which can be directly downloaded as follows:
```
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/AdversarialTabooData.zip
unzip AdversarialTabooData.zip # put the data folder in the main folder
```

# Run the Code
You can simply run the code as follows:
```
python openQA_game.py
python chat_game.py
```

# Citation
If you find the code useful, you can cite our paper:
```
@inproceedings{yao2021adversarial,
 title={Adversarial language games for advanced natural language intelligence},
 author={Yao, Yuan and Zhong, Haoxi and Zhang, Zhengyan and Han, Xu and Wang, Xiaozhi and Zhang, Kai and Xiao, Chaojun and Zeng, Guoyang and Liu, Zhiyuan and Sun, Maosong},
 booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
 pages={14248--14256},
 year={2021}
}
```


