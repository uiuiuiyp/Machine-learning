import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA 

import random
import os
import time

class CollaborativeFilter(object):
    """ Collaborative Filter object """

    def __init__(self, input_directory, n_batch=50, n_feats=5, test_size=0.1, doPCA=False):
        """ Initialize Collaborative Filter object """
        
        self.n_batch   = n_batch
        self.n_feats   = n_feats
        self.test_size = test_size
        self.doPCA     = doPCA

        self._build_input(input_directory)
        self.genres_mat = self._get_genre_feats(input_directory)

        # Initialize the movie and user features (X and Th, respectively)
        self.X  = self._initialize_X()
        self.Th = 0.1 * np.random.randn(self.n_user, self.n_feats) + 1/ np.sqrt(n_feats)

    def _initialize_X(self):

        genre_feature = np.random.randn(self.n_genres, self.n_feats)
        # normalize avg_genre_features so it lies on the surface of (self.n_feats - 1)-d sphere
        genre_feature /= np.sqrt(np.sum(genre_feature ** 2, axis=1)).reshape(-1,1)

        sum_features = self.genres_mat.dot(genre_feature)
        avg_features = sum_features / (self.genres_mat.sum(axis=1).reshape(-1,1) + 1)

        return avg_features

    def _build_input(self, input_directory):
        """ Import ratings and split train and test data set """

        ratings = pd.read_csv(os.path.join(input_directory, "ratings.csv"))

        # Dictionary to convert Ids into row-indices in movie (X) and user (Theta) features
        movieIds     = sorted(list(ratings['movieId'].unique()))
        userIds      = sorted(list(ratings['userId'].unique()))
        self.dict_movieId = dict(zip(movieIds, range(len(movieIds))))
        self.dict_userId  = dict(zip(userIds, range(len(userIds))))

        ratings = ratings.drop('timestamp', axis=1)
        n_ratings = ratings.shape[0]

        ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

        self.train = ratings.iloc[:int((1-self.test_size) * n_ratings)]
        self.test  = ratings.iloc[int((1-self.test_size) * n_ratings):]

        self.n_train = self.train.shape[0]
        self.n_test  = self.test.shape[0]

        self.n_movie = len(movieIds)
        self.n_user  = len(userIds)

        print("num_movie", self.n_movie, "num_user", self.n_user)
    
    def _get_genre_feats(self, input_directory):
        
        """ Generate a genre features that indicate which genre(s)
            each movie belongs to.
        """    
        
        movies = pd.read_csv(os.path.join(input_directory, "movies.csv"))
        movies.genres = movies.genres.apply(lambda g: g.split("|"))

        # Generate a list of all genre mentioned in the movie profile
        all_genre = []
        for index, row in movies.genres.iteritems():
            all_genre = list(set(all_genre + row))
        all_genre.remove('(no genres listed)')
        self.n_genres = len(all_genre)

        for g in all_genre:
            movies[g] = movies.genres.apply(lambda x: g in x)

        genres = movies.drop(['title', 'genres'], axis=1)

        # Drop movies that are not included in the rating.csv
        genres['movieId_row'] = genres.movieId.apply(lambda id: self.dict_movieId.get(id, -1))
        drop_index = genres[genres.movieId_row == -1].index
        genres.drop(drop_index, inplace=True)
        
        return genres.drop(['movieId', 'movieId_row'], axis=1).values.astype(int)

    def _get_batch(self):

        """ 
            Pick batch size number of samples from the training data set
        """
    
        random_indices = [random.randint(0, self.n_train - 1) for _ in range(self.n_batch)]
        batch = self.train.iloc[random_indices]
        return batch.values

    def _gradient_descent(self, X, Th, alpha_decay):
        
        """ Do gradient descent by iterating through the rows of batch
            and optimizing the Movie and User feature
        """

        lamda = 1e-2
        alpha = 0.005 * alpha_decay

        batch = self._get_batch()
        sq_error = 0
        # Would be nice if I don't have to iterate through the rows..
        for row in batch:
            userId, movieId, rating = row

            idx  = self.dict_movieId[movieId]
            idth = self.dict_userId[userId]

            X[idx]   = X[idx]   - alpha * ((X[idx].dot(Th[idth]) - rating) * Th[idth] + lamda * X[idx])
            Th[idth] = Th[idth] - alpha * ((X[idx].dot(Th[idth]) - rating) * X[idx] + lamda * Th[idth])

            sq_error += (X[idx].dot(Th[idth]) - rating) ** 2

        rmse = np.sqrt(sq_error / self.n_batch)
        return rmse
    
    def _get_test_rmse(self):
        sq_error = 0
        for index, row in self.test.iterrows():
            userId, movieId, rating = row

            idx  = self.dict_movieId[movieId]
            idth = self.dict_userId[userId]     

            sq_error += (self.X[idx].dot(self.Th[idth]) - rating) ** 2

        return np.sqrt(sq_error / self.n_test)

    def _nudge(self, alpha_decay):
        
        """ Use the genre feature to 'nudge' the movies that belong to
            the same genres to have similar features.

            To do this:
            1. Generate the average genre feature
            2. Generate average features, in case movies might belong to 
               multiple genres.
                (i.e., if the movie is 'Drama' and 'Comedy',
                the avg_features are 1/2 * Drama_feat + 1/2 * Comedy_feat)
            3. Move the movie toward average feature.

        """
        
        genre_count = np.sum(self.genres_mat, axis=0).reshape(-1,1)
        avg_genre_feature = self.genres_mat.T.dot(self.X) / genre_count

        alpha = 0.01
        sum_features = self.genres_mat.dot(avg_genre_feature)
        avg_features = sum_features / (self.genres_mat.sum(axis=1).reshape(-1,1) + 1)

        self.X = self.X + alpha * (avg_features - self.X)

    def train_model(self, num_epoch=5):

        N_epoch = self.n_train // self.n_batch
        N_total = N_epoch * num_epoch
        N_nudge = 200
        alpha_stages = 25 * N_epoch

        print("number of training sample", self.n_train)
        print("number of test sample", self.n_test)
        print("Total number of steps", N_total)

        start = time.time()

        history = []
        total_rmse = 0
        for i in range(N_total):
            
            alpha_decay = 0.9 ** (i // alpha_stages)

            if (i + 1) % N_epoch == 0:
                print("Epoch {}..".format(i // N_epoch + 1))
                
                avg_train_rmse = total_rmse / N_epoch
                avg_test_rmse  = self._get_test_rmse()
                print("Average training RMSE {:.3f}".format(avg_train_rmse))
                print("Average test RMSE {:.3f}".format(avg_test_rmse))

                history.append([avg_train_rmse, avg_test_rmse])
                total_rmse = 0

            total_rmse += self._gradient_descent(self.X, self.Th, alpha_decay)
            if (i + 1) % N_nudge == 0:
                self._nudge(alpha_decay)

        end = time.time()

        print("Time elapsed to do one epoch {:.3f} sec".format((end - start) / num_epoch))
        print("Done!")

        self.plot_distribution(self.X, "movie", doPCA=self.doPCA)
        self.plot_distribution(self.Th, "user", doPCA=self.doPCA)
        return np.array(history)

    def plot_distribution(self, feature, feat_name, doPCA=False):

        if doPCA:
            pca = PCA()
            feature = pca.fit_transform(feature)
            exp_var = np.sum(pca.explained_variance_ratio_[:2]) * (self.n_feats / 2)
            print(feat_name + " variance ratio:", pca.explained_variance_ratio_)
            title = "Distribution of PCA " + feat_name + " features"
        else:
            exp_var = 1
            title = "Distribution of " + feat_name + " features"

        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_title(title)
        ax.hist2d(feature[:,0], feature[:,1], bins=100, cmap=plt.cm.BuGn_r)
        ax.set_aspect('equal')
        ax.set_title(title)

        plt.savefig(feat_name + "_distribution.png")

def plot_rmse_history(history):

    fig, ax = plt.subplots(figsize=(8,8))
    ax.semilogy(history[:,0], label="Training error")
    ax.semilogy(history[:,1], label="Test error")
    ax.set_title("RMSE error history")
    plt.legend()
    plt.savefig("rmse_history.png")
    
if __name__ == '__main__':
    
    cofi = CollaborativeFilter("ml-latest-small", n_feats=3, test_size=0.1, doPCA=True)
    rmse_history = cofi.train_model(num_epoch=50)
    plot_rmse_history(rmse_history)
