import sys
import os
import pandas as pd
import numpy as np
import pandas as pd
import random
import time
from sklearn import cluster as cl
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from kneed import KneeLocator

RECOM_DIR = './recommendations/'

def load_data(path):
    df = pd.DataFrame()
    if os.path.exists(path):
        df = pd.read_csv(path,)
        return df
    else:
        print('Train data does not exist.')
        return

def reduce_size(df):
    df.drop(labels=['track_href', 'uri', 'Unnamed: 0',
                    'title'], inplace=True, axis=1)

    for column in df:
        if df[column].dtype == 'float64':
            df[column] = pd.to_numeric(df[column], downcast='float')
        if df[column].dtype == 'int64':
            df[column] = pd.to_numeric(df[column], downcast='integer')
    df['genre'] = df['genre'].astype('category')

def normalize_col(df, col):
    df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

def normalize_df(df):
    for col in ['danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms',
                'time_signature', 'key']:
        normalize_col(df, col)

def get_core(df):
    return df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
               'instrumentalness', 'liveness', 'valence', 'tempo', 'genre']]

def reduce_df(df, to):
    dr = SelectKBest(chi2, k=to).fit_transform(
        df.drop(labels=['genre'], axis=1, inplace=False), y=df['genre'])
    return dr

def learn(df_train, df_test):
    min_k = 5
    max_k = 20
    groups_count = range(min_k, max_k+1)
    results = {}
    results['kmeans'] = {'clusters_no': groups_count, 'train_score': [],
                         'test_score': [], 'train_inertia': [], 'duration': [], 'models': [], 'y': []}

    for gc in groups_count:
        st = time.perf_counter()
        kmeans = cl.KMeans(gc)
        y = kmeans.fit_predict(df_train)
        ed = time.perf_counter()
        results['kmeans']['models'].append(kmeans)
        results['kmeans']['y'].append(y)
        results['kmeans']['train_score'].append(
            kmeans.score(df_train)/len(df_train))
        results['kmeans']['test_score'].append(
            kmeans.score(df_test)/len(df_test))
        results['kmeans']['train_inertia'].append(kmeans.inertia_)
        results['kmeans']['duration'].append(ed-st)

    k = KneeLocator(
        groups_count, results['kmeans']['train_inertia'], curve='convex', direction='decreasing')
    best_k = k.elbow-min_k
    final_model = results['kmeans']['models'][best_k]
    final_y = results['kmeans']['y'][best_k]
    return final_model, final_y

def train_model(df) -> None:
    """Preprocessing"""
    reduce_size(df)
    normalize_df(df)
    core = get_core(df)
    df_reduced = reduce_df(core, 5)

    """Processing"""
    df_train, df_test, genre_train, genre_test = train_test_split(
        df_reduced, core['genre'], test_size=0.1, shuffle=True)
    final_model, final_y = learn(df_train, df_test)
    df_labeled = pd.DataFrame({1: df_train[:, 1], 2: df_train[:, 2], 3: df_train[:, 3],
                               4: df_train[:, 4], 0: df_train[:, 0], 'genre': genre_train,
                               'cluster': final_y})
    return final_model, df_labeled

def read_input(args) -> None:
    df_input = pd.DataFrame()
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("User file does not exist")
            sys.exit()
        else:
            df_input = pd.read_csv(file_name)

    return df_input

def distance(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def recommendation(model, df, df_labeled, df_original):
    reduce_size(df)
    normalize_df(df)
    core = get_core(df)
    reduced_input = reduce_df(core, 5)
    prediction = model.predict(reduced_input)
    pred_vals = pd.Series(prediction).value_counts()
    five_topmost_groups = []
    if len(pred_vals) >= 5:
        five_topmost_groups = pred_vals[:5]
    else:
        while len(five_topmost_groups) < 5:
            r = random.random(0, np.unique(df_labeled['cluster']))
            if r not in five_topmost_groups:
                five_topmost_groups.append(r)
    for i, cl in enumerate(five_topmost_groups.index):
        co_clusters = df_labeled[df_labeled['cluster'] == cl]
        df_original.iloc[co_clusters[:5].index].to_csv(
            f'{RECOM_DIR}Daily Mix {i+1}.csv')

    top_songs_at_all = pd.DataFrame()
    centers = model.cluster_centers_
    for c, center in enumerate(centers):
        all_in_cluster = df_labeled[df_labeled['cluster'] == c]
        distances = [{i.name: distance(center, i)} for i in all_in_cluster.drop([
            'genre', 'cluster'], axis=1).iloc]
        distances.sort(key=lambda x: [i for i in x.values()][0])
        closests_to_center = [i for i in [
            distances[0].keys(), distances[1].keys()]]

        top_songs_at_all = top_songs_at_all.append(
            df_original.loc[closests_to_center[0]])
        top_songs_at_all = top_songs_at_all.append(
            df_original.loc[closests_to_center[1]])

    top_songs_at_all.to_csv(f'{RECOM_DIR}top_songs.csv')

def main(args):
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.

    Parameters
    ----------
    args: list 
        list of arguments from the command line
    Returns
    -------
    None
    """

    path = 'genres_v2.csv'
    df_original = load_data(path)
    print("Data is loaded into memory.")
    model, df_labeled = train_model(df_original.copy())
    print("Model is trained.")
    df_input = read_input(args)
    recommendation(model, df_input, df_labeled, df_original)
    print(f"\nYour recommendations are save into '{RECOM_DIR}' directory:")

if __name__ == "__main__":
    args = sys.argv
    main(args)