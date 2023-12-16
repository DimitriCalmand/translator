# French-to-English Translator Learning Project Using Transformers 
# Description
Translatore is an experimental learning project focused on developing a
French-to-English translation tool using Transformer-based models. 
As a work-in-progress, this project aims to explore and demonstrate the 
capabilities of Transformer architectures in the field of natural language 
processing, specifically in language translation.
While the tool is not fully operational or 100% accurate, it serves as an 
educational platform for understanding and experimenting with advanced 
AI techniques in language translation. 

# Installation

### Dataset

#### Dataset Download
Download the necessary dataset for this project from the following link: 
[Europarl Parallel Corpus French-English](https://www.statmt.org/europarl/v7/fr-en.tgz). 
This dataset contains pairs of parallel sentences in French and English, 
extracted from the debates of the European Parliament.

#### Dataset Preparation
Follow the steps below to prepare the dataset for training:

1. **Creating the Dataset**: Run the `create_dataset.py` script to prepare the initial dataset.


``` bash
python3 create_dataset.py
```

2. **Splitting the Dataset**: Use the `split.sh` script to divide the dataset 
into training and testing sets. The specified ratio here is 0.99, 
meaning 99% of the data will be used for training and the rest for testing.

``` bash
./split.sh dataset.csv train.csv test.csv 0.99
```

- The first file, `train.csv`, contains the first 1,987,646 lines for training.
- The second file, `test.csv`, contains the remaining lines for testing.

#### Preparing for Training
Before launching the neural network training, prepare the data:

``` bash
./main tokenizer
```

#### Launching the Neural Network
Once the data is prepared, you can start the training process:

``` bash
./main
```

# Feature
You can change the hyperparameters in the `src/main.py`
``` py 
model = Transformer(
        nb_encoder = 1,
        nb_decoder = 2,
        nb_heads = 4,
        embed_dim = 128,
        feed_forward_dim = 100,
        max_sequence_length = MAX_LENGHT,
        vocab_size = NUM_WORDS
        )
```
also in the src/utils.py
```py
MAX_LENGHT = 30                                                                                         
BATCH_SIZE = 32                                                                 
NUM_WORDS = 10000                                                               
START_WORD = '<start>'                                                          
END_WORD = '<end>'                                                              
NUMBER_WORD = '<number>'                                                        
MAIL_WORD = '<mail>'                                                            
NAME_WORD = '<name>'                                                            

NB_SPECIAL_WORD = 5                                                             
CHUNK_SIZE = 10000
```


# Result

### Source : 
Celui qui a eu des rapports avec l'institution celui qui a présenté des demandes pour participer à un quelconque projet a pu constater la <oov> d'information la <oov> de 
### Prediction :
The one who has had reports with the institution that has submitted to be made for any project to be taken to take part in any project to be 

--------------------------------------------------------------------------------

### Source :
Je souhaite moi aussi une solution rapide de la crise je souhaite moi aussi une présidence forte et je souligne moi aussi que s'il doit y avoir une présidence 
### Prediction :
I also wish to see a rapid solution of the crisis i hope that if there is a presidency and i too hope that if there is a presidency




