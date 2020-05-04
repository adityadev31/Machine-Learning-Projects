import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# read csv file
df = pd.read_csv('movie_dataset.csv')
print(df.head())
print(df.columns)


# list of selected important features
features = ['genres', 'title', 'cast', 'director', 'crew', 'production_companies']


# removinf nulls from each selected feature columns
for x in features:
    df[x] = df[x].fillna('')


# combining and adding those features in a new column
def addFeatures(row):
    try:
        return row['title']+' '+row['cast']+' '+row['genres']+' '+row['director']+' '+row['crew']+' '+row['production_companies']
    except:
        print('error')

df['combined_features'] = df.apply(addFeatures, axis=1)
print(df.combined_features.head())


# initializing countvectorizer
cv = CountVectorizer()
countMatrix = cv.fit_transform(df.combined_features)


# initializing cosine similarity
cos_sim = cosine_similarity(countMatrix)


# supporting functions
def get_title_from_index(x):
    return df[df.index == x]['title'].values[0] # returning title of those data points which have (index==x)
def get_index_from_title(t):
    return df[df.title == t]['index'].values[0] # returning index of those data points which have (title==t)


# user input
user_likes = 'Man on Fire'

# user movie index
user_movie_index = get_index_from_title(user_likes)


# getting list of similar movies to index of user likes movie
similar_movie_list = list(enumerate(cos_sim[user_movie_index]))

# sort list
sorted_list = sorted(similar_movie_list, key=lambda x:x[1], reverse=True)

# printing list
i=0
for item in sorted_list:
    print(get_title_from_index(item[0]))
    i = i+1
    if i>=10:
        break