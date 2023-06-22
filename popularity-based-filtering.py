import pandas


movies_db = pandas.read_csv("movies.csv")
credits_db = pandas.read_csv("credits.csv")
ratings_db = pandas.read_csv("ratings.csv")

# WR = (v / (v + m)) * R + (m / (v + m)) * C
#
# v - vote count
# m - minimum of votes required
# R - average rating
# C - average rating across all movies

m = movies_db["vote_count"].quantile(0.9) # -- take movie above this value
print(m)

C = movies_db["vote_average"].mean() # -- midrange rating of all movies
print(C)

movies_filtered = movies_db.copy().loc[movies_db["vote_count"] >= m]


def weighted_rating(df, m=m, C=C):
    R = df["vote_average"]
    v = df["vote_count"]
    wr = ((v / v+m) * R) + (m / (v + m)* C)
    return wr

# -- creates new column weighted rating
movies_filtered["weighted_rating"] = movies_filtered.apply(weighted_rating, axis=1)
print(movies_filtered)

# -- sort dataset by weighted rating and show first 10
movies_filtered.sort_values("weighted_rating", ascending=False).head(10)
print(movies_filtered.sort_values("weighted_rating", ascending=False).head(10))
