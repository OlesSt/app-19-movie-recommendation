import pandas

movies_db = pandas.read_csv("movies.csv")
credits_db = pandas.read_csv("credits.csv")
ratings_db = pandas.read_csv("ratings.csv")

print(movies_db.head())
print(credits_db.head())
print(ratings_db.head())

