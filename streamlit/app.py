import os

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import numpy as np
#from cdtw import pydtw
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
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

CURRENT_THEME = "light"

#st.set_page_config(layout="wide")

alt.data_transformers.disable_max_rows()  
# to run this:
# streamlit run streamlit/app.py --server.port 8080

# git store token:
# git config --global credential.helper 'cache --timeout=25000'

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


st.title("MADS 695 - Cryptocurrency Analysis")
st.header("Diane O and Oleg N")
st.markdown('\n\n')
st.header("Interactive and supplementary visualizations")
st.markdown('\n\n')

st.markdown(
    "For our Milestone 2 project we had two tasks, supervised learning and unsupervised learning analysis.")
#st.info('Please note each chart below is interactive.')

st.markdown('\n\n')
st.header("Supervised Learning: LSTM, DNN, and RandomForest")


st.write("We had three methods of predicting the next day's price. An LSTM and DNN with pytorch, and a RandomForestRegressor.")

st.subheader("LSTM")

coin = st.selectbox(
     'Select a coin:',
     ('BTC', 'ETH', 'LTC', 'DOGE'))

st.write('You selected:', coin)

filename = str(coin).lower() + '_cm_metrics_final.csv'
orig_df = pd.read_csv(filename)
st.dataframe(orig_df.tail())

orig_df['date'] = pd.to_datetime(orig_df['date'])
plt.figure(figsize = (12, 5))
plt.plot(orig_df['date'], orig_df['PriceUSD'])
plt.title('{} Historical Price'.format(coin), fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price in USD', fontsize=14)

time_index_split = int(np.round((len(orig_df)*.90)))   

plt.axvline(x=orig_df.iloc[time_index_split]['date'] , color='r', label='axvline - full height')
plt.grid()
st.pyplot(plt)
plt.clf()


scaler = MinMaxScaler()
train_split_value = 0.9
num_train = int(np.round(train_split_value * orig_df.shape[0]))

#scaler.fit(orig_df[:num_train][['PriceUSD']])
#normalized_price = scaler.transform(price[['PriceUSD']])
scaler.fit(orig_df[:num_train][['PriceUSD']].values)
normalized_price = pd.DataFrame()
normalized_price['date'] = orig_df['date']
normalized_price['Price'] = scaler.transform(orig_df[['PriceUSD']])

#plot 1:
plt.subplot(1, 2, 1)
plt.plot(orig_df['date'],orig_df['PriceUSD'])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price in USD', fontsize=14)
plt.grid()

#plot 2:
plt.subplot(1, 2, 2)
plt.plot(orig_df['date'],normalized_price['Price'])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Normalized Price', fontsize=14)
plt.grid()
st.pyplot(plt)
plt.clf()


st.code("""
def data_sequences(dataset, target_size):
    d = []
    for idx in range(len(dataset) - target_size):
        d.append(dataset[idx : idx+target_size])
    return np.array(d)

""", language="python")


st.code("""
def split_dataset(dataset, target_size, train_split=0.9):

    data = data_sequences(dataset, target_size)

    num_train = int(np.round(train_split_value * orig_df.shape[0]))

    window_size_adj_split_index = num_train - target_size

    X_train = data[:window_size_adj_split_index, :-1, :]
    y_train = data[:window_size_adj_split_index, -1, :]

    X_test = data[window_size_adj_split_index:, :-1, :]
    y_test = data[window_size_adj_split_index:, -1, :]

    return X_train, y_train, X_test, y_test

""", language="python")

st.code("""
def build_model(X_train, window, num_neurons = 128, activation_function = 'linear'):
    
    model = Sequential([
        LSTM(units=num_neurons, input_shape=(window, X_train.shape[-1]), return_sequences=True),
        LeakyReLU(alpha=0.8),
        LSTM(units=num_neurons, return_sequences=True),
        LeakyReLU(alpha=0.8),
        Dropout(0.2), 
        LSTM(units=int(num_neurons/2), return_sequences=False),
        Dropout(0.2), 
        Dense(units=1, activation=activation_function)
    ])
    return model

""", language="python")


st.code("""
def epochs_loss(model, loss_function = 'mean_squared_error', optimizer='adam'):
    model.compile(
        loss=loss_function, 
        optimizer='adam'
    )
    return model

""", language="python")



csv_header = ['Coin', 'Date', 'Actual', 'Naive', 'window = 1 day', 'window = 7 days', 'window = 30 days', 'window = 90 days', 'window = 200 days']
my_df = pd.read_csv("./streamlit/coin_data/lstm_results.csv", names = csv_header)
my_df.Date = pd.to_datetime(my_df.Date)

df = my_df.copy()
df = df[df['Coin'] == coin]

#st.dataframe(df.head())

plt.figure(figsize = (15,8))
colors = ['green', 'blue', 'red', 'orange', 'purple', 'gray', 'black']

labels = []
for idx, col in enumerate(df.iloc[:,2:].columns):
   plt.plot(df.Date, df[col], color=colors[idx])

   if idx == 0:
       label = col

   else:
       label = col + ", MAE = ${}".format(round(mean_absolute_error(df.Actual, df[col])))

   labels.append(label)    

plt.title('{} Price Prediction'.format(coin), fontsize=14, fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend(labels, loc='best')

plt.grid() 
st.pyplot(plt)


st.subheader("Random Forest Regressor")


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

st.markdown(
    "You can see that there are quite a few columns with nulls. We'll get rid of them before we make normalize.")
st.markdown('\n\n')
st.dataframe(prices_df_orig.head())


st.markdown(
    "To compare and look for similarities between time series, we need to normalize them, which we will do with z-scores. We've removed all columns with nulls. Z-scores are basically the number of standard deviations a value is from a mean.")
st.markdown('\n\n')
st.dataframe(prices_df_z.head())




st.markdown(
    "We will use the z-scores dataframe to make a distance matrix, using dynamic time warping. cdtw from pydtw was the fastest library we found.")
st.markdown('\n\n')

st.code("""
# This is how to actually create the distance matrix

df = squareform(sd.pdist(prices_df_z.T, lambda u, v: pydtw.dtw(u,v,pydtw.Settings(step = 'p0sym', window = 'palival', param = 2.0, norm = False, compute_path = True)).get_dist() )) #~ 10-20  mins")
df_dist = pd.DataFrame(df, columns=prices_df_z.columns, index = prices_df_z.columns)')
""", language="python")


# Reading precomputed distance matrix here
df_dist = pd.read_csv('./streamlit/coin_data/lunar_dist_matrix.csv', index_col = 'Unnamed: 0')

st.dataframe(df_dist.head())

df = df_dist.to_numpy()


st.markdown(
    "This dataframe was precalculated to run on this page.")
st.markdown('\n\n')

st.markdown(
    "Below are a couple ways to sort and see the distance matrix in a slightly organized way. You can start to see two main families of coins by price movement appear.")
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

coin_near_far = st.selectbox(
     'Select a coin:',
     ('ETH-Ethereum', 'BTC-Bitcoin', 'LTC-Litecoin', 'DOGE-Dogecoin'))

st.write('You selected:', coin_near_far)

st.markdown("The top 5 coins closest to " + coin_near_far)
st.dataframe(df_dist[coin_near_far].nsmallest(6).iloc[1:])
st.markdown('\n\n')

nearest_coin = df_dist[coin_near_far].nsmallest(6).index.values[1]


st.markdown("The top 5 coins furthest from " + coin_near_far)
st.dataframe(df_dist[coin_near_far].nlargest(5))
st.markdown('\n\n')

furthest_coin = df_dist[coin_near_far].nlargest(1).index.values[0]

st.markdown("Let's compare some coins and their actual prices to see how close we came.")
plt.clf()
# #plt.rcParams["figure.figsize"] = (20,10)
# plt.figure(figsize=(8,5))
# # multiple line plots
# plt.plot( prices_df_orig.index.values, coin_near_far, data=prices_df_orig, marker='o', markerfacecolor='black', markersize=3, color='black', linewidth=2)
# plt.plot( prices_df_orig.index.values, nearest_coin, data=prices_df_orig, marker='o', markerfacecolor='red', markersize=3, color='red', linewidth=2, secondary_y=True)
# plt.plot( prices_df_orig.index.values, furthest_coin, data=prices_df_orig, marker='o', markerfacecolor='blue', markersize=3, color='blue', linewidth=2, secondary_y=True)

# # show legend
# plt.legend()

# # show graph
# st.pyplot(plt)


# create figure and axis objects with subplots()
plt.figure(figsize=(8,5))
fig,ax = plt.subplots()
# make a plot
ax.plot(prices_df_orig.index.values, coin_near_far, data=prices_df_orig, marker='o', markerfacecolor='black', markersize=3, color='black', linewidth=2)
ax.plot(prices_df_orig.index.values, nearest_coin, data=prices_df_orig, marker='o', markerfacecolor='red', markersize=3, color='red', linewidth=2)
# set x-axis label
ax.set_xlabel("date",fontsize=14)
# set y-axis label
ax.set_ylabel(coin_near_far + ' and ' + nearest_coin,color="red",fontsize=14)

# set xticks rotation before creating ax2
plt.xticks(rotation=45, ha='right')

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
#ax2.plot(prices_df_orig.index.values, nearest_coin, data=prices_df_orig, marker='o', markerfacecolor='red', markersize=3, color='red', linewidth=2)
ax2.plot(prices_df_orig.index.values, furthest_coin, data=prices_df_orig, marker='o', markerfacecolor='blue', markersize=3, color='blue', linewidth=2)
ax2.set_ylabel(furthest_coin,color="blue",fontsize=14)


st.pyplot(plt)






# st.markdown('\n\n')
# st.markdown(
#     "Test")
# st.markdown(
#     "Next is the hourly weather data we used. This is from NOAA, using Central Park as the collection point.\
#         To get it, go ([NOAA](https://www.ncdc.noaa.gov/data-access))  --> Data Access --> Quick Links --> US Local -->\
#              Local Climatological Data (LCD) --> Choose your location(Add to Cart) --> Go to cart at top --> \
#                  LCD CSV, date range --> Continue and give them your email, they'll send it quickly. The documentation is \
#                      [here](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf). ")
# st.markdown('\n\n')
