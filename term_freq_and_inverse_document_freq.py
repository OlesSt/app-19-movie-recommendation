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


def similar_movie(movie_title, nr_movie):
    idx = movies_db.loc[movies_db['title'] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movies_indices = [tpl[0] for tpl in scores[1:nr_movie+1]]
    similar_titles = list(movies_db['title'].iloc[movies_indices])
    return similar_titles


print(similar_movie('Kung Fu Panda 3', 3))

