import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -- Content-Based Filtering --
movies_db = pandas.read_csv("movies_small.csv", sep=";")
print(movies_db)

tfidf = TfidfVectorizer(stop_words='english')
movies_db['overview'] = movies_db['overview'].fillna("")
print(movies_db['overview'])

tfidf_matrix = tfidf.fit_transform(movies_db['overview'])

# -- Similarity matrix --
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
print(similarity_matrix)


