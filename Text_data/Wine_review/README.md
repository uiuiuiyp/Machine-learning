## Wine review

_tags: tf-idf, natural language processing, text mining_

Based on: https://www.kaggle.com/zynicide/wine-reviews

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal of the project is to predict the wine review score based on the wine review descriptions

- We use term frequency-inverse documentary frequency to generate features and use xgboost to fit the model. 

- Importance of words are quantified by examining the "average gain per split"
