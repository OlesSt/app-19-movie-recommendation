import pandas
from surprise import Dataset, Reader
from surprise import SVD
from surprise import model_selection

ratings_db = pandas.read_csv("ratings.csv")[['userId', 'movieId', 'rating']]

# -- create a reader and dataset from database
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(ratings_db, reader)

# -- create a train set, list of values
trainset = dataset.build_full_trainset()

# -- train
svd = SVD()
svd.fit(trainset)

# -- predict rating
print(svd.predict(15, 1956).est)

model_selection.cross_validate(svd, dataset, measures=['RMSE', 'MAE'])