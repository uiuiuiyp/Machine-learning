## Recommendation system

A recommendation system has been coded from scratch. 

Features:
- The original data set is from https://grouplens.org/datasets/movielens/. There are small (~100k ratings) and large (~20M ratings) data sets available on the website.
- Stochastic gradient descent is used to train the _movie_ and _user_ features.
- For each movie, genre information is provided (same movie might belong to multiple genres). The genre information is used to generate _average genre_ movie features and each movie is "nudged" toward the average genre features that it belongs to.

