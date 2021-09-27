import os

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import numpy as np
from cdtw import pydtw
from numpy import savetxt, loadtxt

import scipy.spatial.distance as sd
from scipy.spatial.distance import euclidean, squareform
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from fastcluster import linkage
from datetime import datetime, timedelta
import datetime as dt
from streamlit.proto.Markdown_pb2 import Markdown
from scipy.cluster.hierarchy import fcluster

CURRENT_THEME = "light"

#st.set_page_config(layout="wide")

alt.data_transformers.disable_max_rows()  
# to run this:
# streamlit run streamlit/app.py --server.port 8080

# https://docs.streamlit.io/en/0.65.0/advanced_concepts.html


# https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage


st.title("MADS 692 - Crypto Analysis")
st.header("Diane O and Oleg N")
st.markdown('\n\n')
st.header("Interactive and supplementary visualizations")
st.markdown('\n\n')

st.markdown(
    "This will be an introduction....")
#st.info('Please note each chart below is interactive.')

st.markdown('\n\n')
st.header("Supervised Learning: LSTM, DNN, and RandomForest")








st.header("Unsupervised Learning: Clustering crypto price time series")
st.markdown(
    "We will first look at the original price data. This is two years or crypto price data taken from Lunarcrush.")
st.markdown('\n\n')

prices_df_orig = pd.read_csv('./streamlit/coin_data/lunar_unsup_prices_50.csv', index_col=False)
prices_df_z = pd.read_csv('./streamlit/coin_data/lunar_unsup_prices_z_50.csv', index_col=False)

with st.echo():
    prices_df_orig['time'] = pd.to_datetime(prices_df_orig['time'])
    prices_df_orig = prices_df_orig.set_index('time')

    prices_df_z['time'] = pd.to_datetime(prices_df_z['time'])
    prices_df_z = prices_df_z.set_index('time')
    prices_df_z = prices_df_z.dropna(axis='columns')

st.dataframe(prices_df_z.head())

st.markdown(
    "To compare and look for similarities between time series, we need to normalize them, which we will do with z-scores. We've removed all columns with nulls.")
st.markdown('\n\n')


st.dataframe(prices_df_orig.head())


st.markdown(
    "We will use the z-scores dataframe to make a distance matrix, using dynamic time warping.")
st.markdown('\n\n')
with st.echo():
    # This is how to actually create the distance matrix
    # pydtw was much quicker than others that I found
    
    # df = squareform(sd.pdist(prices_df_z.T, lambda u, v: pydtw.dtw(u,v,pydtw.Settings(step = 'p0sym', window = 'palival', param = 2.0, norm = False, compute_path = True)).get_dist() )) #~ 10-20  mins")
    # df_dist = pd.DataFrame(df, columns=prices_df_z.columns, index = prices_df_z.columns)')

    # Reading precomputed distance matrix here
    df_dist = pd.read_csv('./streamlit/coin_data/lunar_dist_matrix.csv', index_col = 'Unnamed: 0')

st.dataframe(df_dist.head())

df = df_dist.to_numpy()


st.markdown(
    "These calculations are done in real-time, they'll take a couple minutes to run.")
st.markdown('\n\n')

methods = ["ward","single"]#,"average","complete"]
for method in methods:
    
    st.markdown(method)    
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(df,method)
    plt.clf()
    plt.pcolormesh(ordered_dist_mat)
    plt.colorbar()
    plt.xlim([0,len(df)])
    plt.ylim([0,len(df)])
    st.pyplot(plt)
    st.markdown('\n\n')



#plt.rcParams["figure.figsize"] = (100,60)
mat = df_dist
dists = squareform(mat)

st.markdown(
    "Righ-clicking on this image and opening ina new tab will allow you to zoom in closer to inspect the groupings more closely.")
st.markdown('\n\n')

methods = [("ward", 2500)]#("single", 400),("average", 550),("complete", 1300),("weighted", 1050), ("centroid", 400), ("median", 400), ("ward", 2500)]
for method in methods:
    # https://towardsdatascience.com/how-to-apply-hierarchical-clustering-to-time-series-a5fe2a7d8447

    st.markdown(method) 
    plt.clf()
    plt.figure(figsize=(60,10))
    #linkage_matrix = linkage(dists, method[0])
    linkage_matrix = loadtxt('./streamlit/coin_data/dendo_linkage.csv', delimiter=',')#linkage(dists, method[0])
    # hierarchy.dendrogram(linkage_matrix, labels=df_dist.columns, color_threshold=method[1])
    # plt.title("Coin Closeness: " + method[0] + ", cutoff:" + str(method[1]), fontsize=50)
    # plt.xlabel('Coins', fontsize=40)
    # plt.ylabel('Distance', fontsize=30)
    
    # plt.xticks(fontsize= 8)
    # plt.yticks(fontsize=30) # #rotation=90)
    

    # plt.annotate('DOGE and ETC', xy=(2090, 200), xytext=(2500, 5000),
    #         arrowprops=dict(facecolor='black', shrink=0.05, width=7, headwidth=25, headlength=50), fontsize = 50
    #         )
    # plt.annotate('BTC', xy=(1410, 200), xytext=(1100, 4000),
    #         arrowprops=dict(facecolor='black', shrink=0.05, width=7, headwidth=25, headlength=50), fontsize = 50
    #         )

    # plt.annotate('LTC', xy=(275, 200), xytext=(200, 4000),
    #         arrowprops=dict(facecolor='black', shrink=0.05, width=7, headwidth=25, headlength=50), fontsize = 50
    #         )

    # plt.savefig("./streamlit/dendo_" + method[0] + ".png",
    #         bbox_inches ="tight",
    #         dpi=300            
    #         )
    
    
    st.image("./streamlit/dendo_" + method[0] + ".png", width=4000, use_column_width=False)
    st.markdown('\n\n')

num_clusters = []
x_list = list(range(0, 6001, 100))
for i in x_list:
    num_clusters.append(len(np.unique(fcluster(linkage_matrix, i, criterion='distance'))))

print(list(zip(x_list, num_clusters)))

plt.clf()
#plt.rcParams["figure.figsize"] = (10,6)
plt.figure(figsize=(8,5))
plt.xticks(fontsize= 10)
plt.yticks(fontsize=10) # #rotation=90)
plt.plot(x_list, num_clusters)
plt.xlabel("Cutoff")
plt.ylabel("# Clusters")
st.pyplot(plt)


st.markdown("The top 5 coins closest to Bitcoin")
st.dataframe(df_dist['BTC-Bitcoin'].nsmallest(5))
st.markdown('\n\n')


st.markdown("The top 5 coins furthest from Bitcoin")
st.dataframe(df_dist['BTC-Bitcoin'].nlargest(5))
st.markdown('\n\n')

st.markdown("Let's compare some coins and their actual prices to see how close we came.")
plt.clf()
#plt.rcParams["figure.figsize"] = (20,10)
plt.figure(figsize=(8,5))
# multiple line plots
#plt.plot( prices_df_orig.index.values, 'BTC-Bitcoin', data=prices_df_orig, marker='o', markerfacecolor='blue', markersize=3, color='skyblue', linewidth=2)
plt.plot( prices_df_orig.index.values, 'APR-APR Coin', data=prices_df_orig, marker='o', markerfacecolor='red', markersize=3, color='red', linewidth=2)
plt.plot( prices_df_orig.index.values, 'DGTX-Digitex Futures', data=prices_df_orig, marker='o', markerfacecolor='blue', markersize=3, color='blue', linewidth=2)

# show legend
plt.legend()

# show graph
st.pyplot(plt)

st.markdown('\n\n')
st.markdown(
    "Test")
st.markdown(
    "Next is the hourly weather data we used. This is from NOAA, using Central Park as the collection point.\
        To get it, go ([NOAA](https://www.ncdc.noaa.gov/data-access))  --> Data Access --> Quick Links --> US Local -->\
             Local Climatological Data (LCD) --> Choose your location(Add to Cart) --> Go to cart at top --> \
                 LCD CSV, date range --> Continue and give them your email, they'll send it quickly. The documentation is \
                     [here](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf). ")
st.markdown('\n\n')
