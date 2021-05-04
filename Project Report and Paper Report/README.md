# FENCE: Fake News Classifier

To classify fake news data over different domains across different types of data (News Article Headlines and Tweets)

## Requirements 

- appdirs
- click
- funcsigs
- gunicorn
- h5py
- itsdangerous
- Jinja2
- Keras
- MarkupSafe
- mock
- nltk
- numpy
- packaging
- pbr
- protobuf
- pyparsing
- PyYAML
- scipy
- six
- tensorflow
- Theano
- Werkzeug
- tqdm
- spacy
- textstat
- nltk
- subprocess
- tensorflow-hub

Further details about the specific version of the package is mentioned in the requirements.txt file.

## Getting Started

1. Install all the requirements from requirements.txt file. 

       pip install -r requirements.txt

2. The dataset is stored in the data folder. 

3. Download as zip the clickbait detector repository (link in refernces) and copy its files in the same location as the EDA.py file.

4. To perform EDA analysis and obtain the dataset with EDA analysis as features

       python dataset.py

5. To train the model and perform analysis over types and different domains of data 

       python siamesenetwork.py

6. The Project report is stored in the Report folder.

## References

- [ClickBait Detector](https://github.com/saurabhmathur96/clickbait-detector)
- [Redability Index](https://www.geeksforgeeks.org/readability-index-pythonnlp/)
- [One Shot Learning](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
- [Stanford POS Tagger Tutorial](https://www.linguisticsweb.org/doku.phpid=linguisticsweb:tutorials:linguistics_tutorials:automaticannotation:stanford_pos_tagger_python)
- [Triplet Loss](https://www.youtube.com/watch?v=d2XB5-tuCWU)


