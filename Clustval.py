# Compare Silhouette coefficients of different clustering algorithms at different number of clusters
import pandas as pd
import numpy as np
import wget
import plotly.express as px
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Function to load data
def load(url):
    df = pd.read_csv(wget.download(url))
    return df

# Function to subset data
def subset(df,gen):
    df1 = df[["Genotype","Repeat","HT_Conc_uM","Time_m","normed_intensity_ch2"]]
    df1 = df1[df1["Genotype"] == gen] [df1["Time_m"]>0].sort_values(by = ["HT_Conc_uM","Time_m"])
    return df1

# Function to reshape data
def dat(df):
    dat = np.asarray(df['normed_intensity_ch2'],df['normed_intensity_ch2']).reshape(df['normed_intensity_ch2'].size, -1)
    return dat

# Function to calculate range of cluster number needed for screening
def ncluster(df):
    time =  len(df['Time_m'].unique())
    conc = len(df['HT_Conc_uM'].unique())
    r = time*conc # r is the number of clusters needed for screening = number of timepoints * number of HT concentrations
    return r

# Function to calculate Silhouette coefficients for each number of clusters with K-means algorithm
def ktest(df):
    data = dat(df)
    k = []
    n  = []
    for i in range(2,ncluster(df)):
        kmeans = KMeans(n_clusters = i)
        klabels  = kmeans.fit_predict(data)
        k = np.append(k,silhouette_score(data, klabels))
        n = np.append(n,i)
    results = pd.DataFrame(k,n )
    plot = px.scatter(results,
                              title = "Silhouette coeffcicient as a function of cluster number with K-means clustering",
                              template ="plotly_dark")
    plot.update_xaxes(title = "n cluster")
    plot.update_yaxes(title = "Silhouette coeffcient" )
    plot.show(renderer = "vscode")
    print(f"Best Silhouette coefficient is {results.max()}\n")
    print(f"With {results.idxmax()} clusters")
    return results

# Function to calculate Silhouette coefficients for each number of clusters with Agglomerative clustering algorithm
def aggtest(df):
    data = dat(df)
    k = []
    n  = []
    for i in range(2,ncluster(df)):
        aggclust = AgglomerativeClustering(n_clusters = i,linkage = 'complete')
        klabels  = aggclust.fit_predict(data)
        k = np.append(k,silhouette_score(data, klabels))
        n = np.append(n,i)
    results = pd.DataFrame(k,n )
    plot = px.scatter(results,
                              title = "Silhouette coeffcicient as a function of cluster number with Agglomerative clustering",
                              template="plotly_dark")
    plot.update_xaxes(title = "n cluster")
    plot.update_yaxes(title = "Silhouette coeffcient" )
    plot.show(renderer = "vscode")
    print(f"Best Silhouette coefficient is {results.max()}\n")
    print(f"With {results.idxmax()} clusters")
    return results

# Calculate Silhouette coefficient for Mean Shift algorithm
def mtest(df):
    data = dat(df)
    k = []
    mshift = MeanShift()
    klabels  = mshift.fit_predict(data)
    k = np.append(k,silhouette_score(data, klabels))
    print(f"Silhoutte coeffecient for Mean shift clustering is{k}\n")
    print(f"With cluster centers:{mshift.fit(data).cluster_centers_} ")
    return k

if __name__ == "__main__":
    df_acc = load("https://www.dropbox.com/s/di568r38yhduavw/HT_accumulation_demo.csv?dl=1")
    df_accWT = subset(df_acc, "WT")
    df_accTolC = subset(df_acc,"DTolC")
    with joblib.parallel_backend('loky'):
        mwt = mtest(df_accWT)
        kwt = ktest(df_accWT)
        hwt = aggtest(df_accWT)
        mtolc = mtest(df_accTolC)
        ktolc = ktest(df_accTolC)
        htolc = aggtest(df_accTolC)
       

