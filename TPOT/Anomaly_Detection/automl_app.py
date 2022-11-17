import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import glob
import os
import random


"""Read Input Files"""

path = r'/Users/akshayshembekar/Downloads/input'    # Set location where your files are
all_files = glob.glob(os.path.join(path, "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, encoding='cp1252')
    li.append(df)

df_data = pd.concat(li, axis=0, ignore_index=True)
df_data.columns = df_data.columns.str.lower()

"""Plot Histogram"""
df_data['Time_Formatted'] = pd.to_datetime(df_data['Time'])
df_hist = df_data[['Time_Formatted']]
df_hist.rename(columns={'Time_Formatted': 'Time_24hr'})
df_hist.groupby(df_hist["Time_24hr"].dt.hour).count().plot(kind="bar")


print(df_data.shape)

"""Apply Classifier Rules"""

df_data['anomaly'] = 0
df_data.loc[df_data['protocol'] == 'NBNS', 'anomaly'] = 1
df_data.loc[df_data['protocol'] == 'SSDP', 'anomaly'] = 1
df_data.loc[df_data['protocol'] == 'TELNET', 'anomaly'] = 1
df_data.loc[(df_data['protocol'] == 'FTP') & (df_data['length'] >= 1000), 'anomaly'] = 1


"""Get Geo-Location of IPs"""
#Pass

print(df_data.shape)

profile_post = ProfileReport(df_data, title="Anomaly Detection Profiling Report")
profile_post.to_file("Network_Anomaly_Detection_EDA.html")

df_data.drop_duplicates(inplace=True)
df_data = df_data.drop(columns=['no.', 'time', 'source', 'destination', 'info'])
df_data = pd.get_dummies(df_data, prefix=['protocol'],  columns=['protocol']).reset_index()
X = df_data.drop(['anomaly'], axis=1)
y = df_data['anomaly']
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df_data['anomaly'], test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
pipeline_optimizer = TPOTClassifier(generations=5, population_size=50, cv=cv,  scoring='accuracy',
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
