from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, roc_curve
 
import matplotlib.pyplot as plt
 
FIG_PATH = './figures/'

def distance(x, y):
    dist = 0
    for col in [0, 1, 2, 3, 4]:
        if x[col] != y[col]:
            dist += 1
    for col in [5]:
        dist += abs(x[col]-y[col])
    return dist

if __name__ == '__main__':
    df = pd.read_csv('data/sample.csv')
    one = df[df['is_attributed']==1]
    zero = df[df['is_attributed']==0].sample(len(one))

    df = pd.concat([one,zero]).sample(5000)

    df.drop(columns=['Unnamed: 0'], inplace=True)

    """Preprocessing"""
    df.drop(['attributed_time'], axis=1, inplace=True)
    for col in ['ip', 'app', 'device', 'os', 'channel']:
        df[col] = df[col].astype('category')

    df['is_attributed'] = df['is_attributed'].astype('bool')

    for col in ['click_time']:
        df[col] = pd.to_datetime(
            df[col], format='%Y%m%d %H:%M:%S').astype(int) / 10**9

    df['click_time'] = df['click_time']-df['click_time'].min()
    df['click_time'] = df['click_time']/df['click_time'].max()

    """Sampling"""
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['is_attributed'],
                                                                inplace=False, axis=1), df['is_attributed'], test_size=0.2)

    df_models = pd.DataFrame(
        columns=['model_name', 'model', 'accuracy', 'time'])

    """Testing models"""
    """Test knn"""
    st = time.perf_counter()
    neigh = KNeighborsClassifier(n_neighbors=5, metric=distance)
    neigh.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': 'KNN', 'model': neigh, 'accuracy': neigh.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('knn done.')

    """Test Logistic Regression"""
    st = time.perf_counter()
    lgs = LogisticRegression(random_state=0)
    lgs.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': "Logistic Regression", 'model': lgs, 'accuracy': lgs.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('logistic regression done.')

    """Test Decision Tree"""
    st = time.perf_counter()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': "Decision Tree", 'model': dt, 'accuracy': dt.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('decision tree done.')

    """Test Random Forest"""
    st = time.perf_counter()
    raf = RandomForestClassifier(max_depth=2, random_state=0)
    raf.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': "Random Forest", 'model': raf, 'accuracy': raf.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('random forest done.')

    """Test SVM"""
    st = time.perf_counter()
    sv = svm.SVC(probability=True)
    sv.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': "SVM", 'model': sv, 'accuracy': sv.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('svm done.')

    """Test Naive Bayes"""
    st = time.perf_counter()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    ed = time.perf_counter()
    df_models = df_models.append({'model_name': "Naive Bayes", 'model': nb, 'accuracy': nb.score(
        X_test, y_test)*100, 'time': ed-st}, ignore_index=True)
    print('naive bayes done.')

    """Reporting"""
    if not os.path.exists(FIG_PATH):
        os.makedirs(FIG_PATH)
    plt.rcParams["figure.figsize"] = (6, 5)
    df_models['acc/time'] = df_models['accuracy']/np.sqrt(df_models['time'])
    print(df_models[['model_name', 'accuracy', 'time', 'acc/time']])
    df_models[['model_name', 'accuracy', 'time',
               'acc/time']].round(3).to_csv('results.csv')

    for i in df_models.iloc:
        """Heatmap"""
        model = i['model']
        model_name = i['model_name']
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt=".2%", linewidths=.5)
        plt.xlabel('Real labels')
        plt.ylabel('Predicted labels')
        plt.title(f"{model_name} Confusion matrix\n")
        fname = FIG_PATH+f"{model_name}-heatmap.png"
        plt.savefig(fname)
        print(f'plot saved at {fname}')
        plt.close()

        """ROC Curve"""
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(f'ROC curve for {model_name}')
        plt.close()

    plt.rcParams["figure.figsize"] = (8, 6)
    

    sns.barplot(data=df_models, x='model_name', y='accuracy')
    fname = FIG_PATH+"compare_barplot.png"
    plt.savefig(fname)
    print(f'plot saved at {fname}')
    plt.close()

