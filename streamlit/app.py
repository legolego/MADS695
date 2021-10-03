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
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, median_absolute_error, accuracy_score, max_error, classification_report, confusion_matrix


CURRENT_THEME = "light"

#st.set_page_config(layout="wide")

alt.data_transformers.disable_max_rows()  
# to run this:
# streamlit run streamlit/app.py --server.port 8080
# "top" and then "kill -9 PID"

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
    """For our Milestone 2 project we had two tasks, supervised learning and unsupervised learning analysis. 
    We decided to try to do next-day price prediction for the supervised learning part because it was a simpler goal for us to understand. 
    For the unsupervised learning part of the project, we wanted to see if we could find families of coins whose prices tend to move together. 
    This was accomplished by finding similar price movement through dynamic time warping and clustering, with the result being quite reasonable. """)
#st.info('Please note each chart below is interactive.')

st.markdown(
    """For data collection, we used Coinmetrics and LunarCrush, which cover over a thousand cryptocurrencies providing real-time, daily & weekly updates and historical data,
    regarding market volume, market capitalization, transactions, coins in wallet, social media data, etc. 
    Coinmetrics archives cryptocurrencies since January 2010, meaning data are available almost from creation for almost all existing coins. 
    LunarCrush makes available the last two years worth of data, mostly having to do with social media. 
    Both Coinmetrics and LunarCrush provide free APIs that facilitate historical data downloading."""
)

#st.write("We had two methods of predicting the next day's price. An LSTM with pytorch, and a RandomForestRegressor.")

st.markdown('\n\n')
st.header("Supervised Learning: LSTM and RandomForest")

st.write("""Our supervised learning objective is to predict the future price for cryptocurrencies. 
    We predict the daily close price of a given coin at time t, based on previous daily close prices from t-T to t-1 for the coin in question. 
    This is a univariate time series with a training window of size T. We had two methods of predicting the next day's price. An LSTM with pytorch, and a RandomForestRegressor.""")

st.subheader("LSTM")

st.markdown(
    """
    This section walks through the implementation of a recurrent neural network using Long Short-term Memory (LSTM) layers to predict future cryptocurrency values based on historical prices. 
    LSTM is actually recognized for such time series forecasting applications. 
    The method is known for maintaining an internal state that keeps track of the data previously seen, thereby providing a convenient way to model long-term dependencies. 
    LSTM requires a 3D tensor input with batch size, timesteps (i.e. T), and input dimension equals 1 in our case since we first focus on a single feature (i.e. price). 
    Let's start by selecting a coin..."""
)

coin = st.selectbox(
     'Select a coin:',
     ('BTC', 'ETH', 'LTC', 'DOGE'))

st.write(" You selected: " + str(coin) + ". The table below provides a few observations from the original data, which was retrieved from Coinmetrics. (You may want to scroll to the right to see all coin attributes.)")  


filename = str(coin).lower() + '_cm_metrics_final.csv'
orig_df = pd.read_csv(filename)
st.dataframe(orig_df.tail())


st.write('As earlier stated, we essentially focus on coin prices for the supervized learning study. The plot below provides a view of ' + str(coin) + ' historical prices extracted from "PriceUSD" column in the above-mentioned dataframe, along with the cutoff for 90% of the data going to the training set.')

orig_df['date'] = pd.to_datetime(orig_df['date'])
plt.figure(figsize = (13, 6))
plt.plot(orig_df['date'], orig_df['PriceUSD'], linewidth=3.0)
plt.title('{} Historical Price'.format(coin), fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price in USD', fontsize=14)

time_index_split = int(np.round((len(orig_df)*.90)))   

plt.axvline(x=orig_df.iloc[time_index_split]['date'], linewidth=3.0, color='r', label='axvline - full height')
plt.grid()
st.pyplot(plt)
plt.clf()

    # Data also requires some preprocessing prior to LSTM application. 
    # Preliminary clean up tasks include deleting nan observations and sorting data in ascending order by date; 
    # this ensures predictions only use past data to predict the future. 
    # Coins are so volatile that it is also safe to normalize price values, which is particularly helpful from LSTM computation perspective. 
    # The key with normalization is to fit the scaler to training data only, before transforming the entire dataset. 
    # This mimics reality where we don’t have a clue about the future. 

st.write("""
            Historical prices highlight a broad range of values that might not be manageable or at least computational exhaustive when running LSTM, so the first step is to normalize data. 
            We decided to bring values back to the interval [0,1], fitting the scaler to training data only. 
    """)

if coin != 'LTC':
    st.write(" Note the values above 1 when we transform the entire dataset; this is due to the strike observed early 2021 in the historical price that appears in test data only.")

scaler = MinMaxScaler()
train_split_value = 0.9
num_train = int(np.round(train_split_value * orig_df.shape[0]))

#scaler.fit(orig_df[:num_train][['PriceUSD']])
#normalized_price = scaler.transform(price[['PriceUSD']])
scaler.fit(orig_df[:num_train][['PriceUSD']].values)
normalized_price = pd.DataFrame()
normalized_price['date'] = orig_df['date']
normalized_price['Price'] = scaler.transform(orig_df[['PriceUSD']])


plt.figure(figsize = (13, 6))
plt.title('Actual vs Normalized Prices', fontsize=14, fontweight='bold')
#plot 1:
plt.subplot(1, 2, 1)
plt.plot(orig_df['date'],orig_df['PriceUSD'], linewidth=3.0)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price in USD', fontsize=14)
plt.grid()

#plot 2:
plt.subplot(1, 2, 2)
plt.plot(orig_df['date'],normalized_price['Price'], linewidth=3.0)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Normalized Price', fontsize=14)
plt.grid()
st.pyplot(plt)
plt.clf()

st.write("""
    Another important step of data preprocessing is to reorganize the data in a way that a sequence of the values in previous T days is used to predict the value at time t. 
    This is done by recording every T+1 consecutive prices, starting at 0 and incrementing by 1 until the (T+1)th last (i.e. from the end); 
    then a historical array keeps the first T elements and a target vector records the last element from the T+1 sequence under consideration. 
    These are then split for training and testing.
""")
st.code("""
def data_sequences(dataset, target_size):
    d = []
    for idx in range(len(dataset) - target_size):
        d.append(dataset[idx : idx+target_size])
    return np.array(d)

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


st.write("""
    At this stage, we use the tensorflow Python library to build a neural network of 3 LSTM and 1 dense layers.  
    We compile the model using a tensorflow “adam” optimizer, recommended in literature for regression tasks, and a mean squared error (MSE) loss function. 
    The last step is to fit the model. We mainly use hyper parameters available in the literature, with light tuning based on observations. 
""")

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

def epochs_loss(model, loss_function = 'mean_squared_error', optimizer='adam'):
    model.compile(
        loss=loss_function, 
        optimizer='adam'
    )
    return model

""", language="python")


st.write("""
    We can now run the model and make predictions.
""")

st.code("""
window = 90
X_train, y_train, X_test, y_test = split_dataset(normalized_price[['Price']], target_size = window + 1, train_split=train_split_value) 
model = build_model(X_train, window, num_neurons = 128, activation_function = 'linear')
model = epochs_loss(model, loss_function = 'mean_squared_error', optimizer='adam')
print(model.summary())

history = model.fit(
    X_train, 
    y_train, 
    epochs=25, 
    batch_size=64, 
    shuffle=False,
    validation_split=0.1
)

model.predict(X_test)
    
""", language="python")

st.write("""
    The model effectiveness is guaranteed by MSE learning curves. Below is an example for a window size of 200 days; except for Ethereum, you can see how our model converges (typically in less than 15 epochs).
""")

csv_header = ['epoch', 'loss', 'val_loss', 'coin']
my_df = pd.read_csv("./streamlit/coin_data/lstm_mse_200.csv", names = csv_header)
my_df = my_df[my_df['coin'] == coin].reset_index()

plt.figure(figsize = (12,5))
plt.plot(my_df['loss'], linewidth=3.0)
plt.plot(my_df['val_loss'], linewidth=3.0)
plt.title('Learning Curve, window = 200', fontsize=16, fontweight='bold')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid() 
st.pyplot(plt)

st.write("Finally, the plots below provide " + str(coin) + "predictions for multiple window sizes. ")
st.write("""
predictions follow the same trend as the actual price values. However, except for the naive prediction model that just assigns t-1 price value to t, the other attempts show a higher mean absolute error (MAE) and give the impression that there is a cap that predictions cannot overtake. MAE was chosen for its simple interpretability, as it has the units of the dependent variable. Even predictions for a 1 day window that could be expected to be similar to the naive prediction screw up during high spikes. In our opinion, this is caused by the nature of the data and the normalization scaler that is set on the basis of training data only, where it is not possible to anticipate the spike observed in mid 2020. 
"""
)

st.write(
"""
Expanding the training set to encompass this might create other issues because this will represent 96 to 97% of the data; using only 3% of data for validation would certainly make MAE noisy. Another idea to fix the scaler was to add a very big, temporary value when fitting (e.g. $80000); this didn’t fix the issue but playing with larger windows that could maintain in memory spikes (even much smaller) from the past until current prediction cases, and other coins where similar peaks had been experienced in training data greatly improved MAE (i.e. approximately $2000 for 200 days) and almost perfect predictions for LTC. The reader would certainly want to refer to the notebook for those improvements.
"""
)

st.write(
"""
DOGE coin prediction was a complete failure. The problem here is not our model but in the data; looking at the original dataset, we see that there is a total break with the past. Nothing similar to the future is observed in the past and sent to the model.
"""
)

csv_header = ['Coin', 'Date', 'Actual', 'Naive', 'window = 1 day', 'window = 7 days', 'window = 30 days', 'window = 90 days', 'window = 200 days']
my_df = pd.read_csv("./streamlit/coin_data/lstm_results.csv", names = csv_header)
my_df.Date = pd.to_datetime(my_df.Date)

df = my_df.copy()
df = df[df['Coin'] == coin]

#st.dataframe(df.head())

plt.figure(figsize = (16,10))
colors = ['green', 'blue', 'red', 'orange', 'purple', 'gray', 'black']

labels = []
for idx, col in enumerate(df.iloc[:,2:].columns):
   plt.plot(df.Date, df[col], linewidth=3.0, color=colors[idx])

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

st.write("PLEASE FEEL FREE TO CHANGE COIN SELECTION AND SEE RESULTS FOR OTHER CRYPTOCURRENCIES.")


st.header("Random Forest Regressor")

st.write("Let's pick one of the two different data sources we have. Coinmetrics provides more financial data, and information about coin wallets, while LunarCrush  has social media data related to each coin.")

which = st.selectbox(
     'Select a source:',
     ('Coinmetrics', 'Lunar Crush'))

st.write(" You selected: " + str(which) + ". ")  

if which == 'Coinmetrics':
    df_cm = pd.read_csv('./btc_cm_metrics_final.csv', index_col=False)
    df_cm['date'] = df_cm['date'].str[:10]
    df_cm['date'] = pd.to_datetime(df_cm['date'])
    df_cm = df_cm.set_index('date')

elif which == 'Lunar Crush':
    df_cm = pd.read_csv('./btc_vals_lunar_social.csv', index_col=False)
    df_cm = df_cm.drop('asset_id', axis=1)
    df_cm['time'] = pd.to_datetime(df_cm['time'])
    df_cm.rename(columns={'close': 'PriceUSD', 'time': 'date'}, inplace=True)
    df_cm = df_cm.set_index('date')

st.dataframe(df_cm.head())

st.write("We'll get rid of all rows where PriceUSD is nan, then drop all other columns thathave any nans in them. We know we have a complete values for all remining columns now. We will make acouple columns that just shift past values into the future one or two days, but we'll leave the nans that result.")


with st.echo():
    df_cm = df_cm.dropna(subset=['PriceUSD'])            # drop all rows where priceUSD is NAN
    df_cm = df_cm.dropna(axis=1)                         # drop all remaining columns with NANs now, since Price is our target
    df_cm['close_1'] = df_cm['PriceUSD'].shift(-1)
    df_cm['close_2'] = df_cm['PriceUSD'].shift(-2)       # our targets, 1 or 2 days ahead


pct_to_train_with = st.selectbox(
     'Select a % split for the training/test sets:',
     (.8, .85, .9, .95, .97, .98, .99))

st.write(" You selected: " + str(pct_to_train_with) + ". ")

# split test/train

train_days = int(np.round((len(df_cm)*pct_to_train_with)))
test_days = len(df_cm) - train_days


train_df = df_cm.head(train_days)
test_df = df_cm.tail(test_days)

start_date_of_train = pd.to_datetime(train_df.index[0].strftime('%Y-%m-%d'))
start_date_of_test = pd.to_datetime(test_df.index[0].strftime('%Y-%m-%d'))

st.write("We're left with " + str(train_days) + " training days, and " + str(test_days) + " test days." )

st.write("Next, depending on the chose data set, we'll exclude columns from the regressor. They're either our target columns, or too closely correlated with them, having already seen their coeffecients after running previous regressions." )
with st.echo():
    # exclude if only using Coin Metrics
    # These are too correlated with price and were manually removed after looking at feature importance
    # Market Cap numbers should definately be removed
    # our target also should be in the training data

    if which == 'Coinmetrics':
        features_to_exclude = ['close_1', 'close_2', 'CapAct1yrUSD', 'CapMVRVCur',\
                            'CapMVRVFF', 'CapMrktCurUSD', 'CapMrktFFUSD', 'CapRealUSD', 'PriceBTC', 'PriceUSD']
    elif which == 'Lunar Crush':
        # exclude from Lunar Crush
        features_to_exclude = ['PriceUSD', 'close_1', 'close_2', 'high', 'low',\
                            'open', 'market_cap', 'market_cap_global']

    features = list(train_df.drop(features_to_exclude, axis=1).columns)


days_to_predict = st.selectbox(
     'Select the number of days to predict ahead:',
     (1, 2))

st.write(" You selected: " + str(days_to_predict) + ". ")

st.write("Let's make our test and training datasets.")

with st.echo():
    X_train = train_df[features]
    y_train = train_df['close_' + str(days_to_predict)]

    X_test = test_df[features].iloc[:-days_to_predict]
    y_test = test_df['close_' + str(days_to_predict)].iloc[:-days_to_predict]

st.write("We ran a grid search, but it's values weren't optimal, so we had to tweak them manually.")

with st.echo():
    # param_grid = {'n_estimators': [75, 95, 110, 125, 135],
    #               'max_depth': [2, 3, 4, 5, 6],
    #               "max_features": ['log2', 'auto', 'sqrt'],          # log2 was best previously
    #               "criterion": ['mae']} # 'mse',                       # mae was best previously
    # scorers = {
    #     'mse': make_scorer(mean_squared_error),
    #     'max_error' : make_scorer(max_error),
    #     'mae' : make_scorer(median_absolute_error)
    # }

    # regr = RandomForestRegressor(random_state=42)
    # grid = GridSearchCV(regr, param_grid, cv=5, refit='mae', return_train_score=True, scoring=scorers) 
    
    # grid.fit(X_train, y_train) 
    
    # print(grid.best_params_) 
    # grid_predictions = grid.predict(X_test) 

    # found by gridsearch for Coin Metrics
    # {'criterion': 'mae', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 110}

    # found by gridsearch for LunarCrush
    # {'criterion': 'mae', 'max_depth': 2, 'max_features': 'auto', 'n_estimators': 95}

    #opt_params = grid.best_params_

    if which == 'Coinmetrics':
        # Coin Metrics
        opt_params = {'criterion': 'mae', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 100}
    elif which == 'Lunar Crush':
        # LunarCrush
        opt_params = {'criterion': 'mae', 'max_depth': 11, 'max_features': 'auto', 'n_estimators': 130}

st.write("We'll call the regressor, unpacking the parameters in the dictionary above. This may take a few seconds, the regressor is actually running right now.")
with st.echo():
    regr = RandomForestRegressor(random_state=42, **opt_params)
    regr.fit(X_train, y_train)

st.write("Some regressors give you feature importance for free, and it's valueable to look at these values to see which are most influential. We'll look at the Top 5 here.")


feats = [x for _, x in sorted(zip(regr.feature_importances_,features),reverse=True)]
importance = sorted(regr.feature_importances_,reverse=True)

st.write(list(zip(feats, importance))[:5])

dfPlot = pd.DataFrame(list(zip(feats, importance)),
               columns =['Features', 'Importance'])
dfPlot['CumSum'] = dfPlot['Importance'].cumsum()

st.write("Next we'll make a Pareto chart with those features that together make up 90% of the importance out of all of them.")

# Get features that make up at least this percentage of importance according to the regressor
thresh = .9
thresh_len = len([x for x in np.cumsum(sorted(regr.feature_importances_,reverse=True)) if x <= thresh])

# https://medium.com/analytics-vidhya/creating-a-dual-axis-pareto-chart-in-altair-e3673107dd14
sort_order = dfPlot["Features"].tolist()
# The base element adds data (the dataframe) to the Chart
# The categories of complaints are positioned along the X axis
base = alt.Chart(dfPlot.head(thresh_len+1), title="Pareto chart for top " + str(thresh_len+1) + " features from " + which).encode(
    x = alt.X("Features:O",sort=sort_order),
).properties (
width = 600, 
height = 500
)
# Create the bars with length encoded along the Y axis 
bars = base.mark_bar(size = 30).encode(
    y = alt.Y("Importance:Q"),
).properties (
width = 600
)
# Create the line chart with length encoded along the Y axis
line = base.mark_line(
                       strokeWidth= 1.5,
                       color = "#cb4154" 
).encode(
    y=alt.Y('CumSum:Q',
             title='Cumulative Importance',
             axis=alt.Axis(format=".0%")   ),
    text = alt.Text('CumSum:Q')
)
# Mark the percentage values on the line with Circle marks
points = base.mark_circle(
              strokeWidth= 3,
              color = "#cb4154" 
).encode(
         y=alt.Y('CumSum:Q', axis=None),
)
# Mark the bar marks with the value text
bar_text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=-10,  #the dx and dy can be manipulated to position text
    dy = -10, #relative to the bar
).encode(
    y= alt.Y('Importance:Q', axis=None),
    # we'll use the percentage as the text
    text=alt.Text('Importance:Q', format="0.2f"),
    color= alt.value("#000000")
)

# Mark the Circle marks with the value text
point_text = points.mark_text(
    align='left',
    baseline='middle',
    dx=-10, 
    dy = -10,
).encode(
    y= alt.Y('CumSum:Q', axis=None),
    # we'll use the percentage as the text
    text=alt.Text('CumSum:Q', format="0.0%"),
    color= alt.value("#cb4154")
)
# Layer all the elements together 
st.altair_chart((bars + line + bar_text ).resolve_scale(
    y = 'independent'
))


st.write("We will rerun the regression again, this time with the fewer features we found and show the results in a plot, as well as an MAE score.")

# Including all features under importance threshold
updated_feats = feats[:thresh_len+1]

print(updated_feats)

# Re-training model with smaller subset of features
regr = RandomForestRegressor(random_state=42, **opt_params)


X_train = train_df[updated_feats]
y_train = train_df['close_' + str(days_to_predict)]

regr.fit(X_train, y_train)

feat_importance = regr.feature_importances_

feats = [x for _, x in sorted(zip(feat_importance,updated_feats),reverse=True)]
importances = sorted(feat_importance,reverse=True)

train_predictions = regr.predict(X_train) # X has new updated top feature list

date_first_pred_train = start_date_of_train + timedelta(days=days_to_predict)

df_preds_train = pd.DataFrame(zip(pd.date_range(date_first_pred_train, periods=len(train_predictions)).tolist(), train_predictions ))
df_preds_train.columns = ['date', 'train_pred']

X_test = test_df[updated_feats]

test_predictions = regr.predict(X_test)

date_first_pred_test = start_date_of_test + timedelta(days=days_to_predict)

df_preds_test = pd.DataFrame(zip(pd.date_range(date_first_pred_test, periods=len(test_predictions)).tolist(), test_predictions ))
df_preds_test.columns = ['date', 'test_pred']

scale = alt.Scale(domain=['Actual Price', 'Predicted Training Price', 'Predicted Test Price'], range=['lightgreen', 'blue', 'orange'])

bdf = df_cm.reset_index()[['date', 'PriceUSD']]

bdf.insert(0, 'ColVal', 'Actual Price')
with st.echo():
    actualBTC = alt.Chart(bdf, title="BTC Prediction for " + str(days_to_predict) + " days ahead with " + str(pct_to_train_with) + " Training Set").mark_line().encode(
        x='date:T',
        y='PriceUSD:Q',
        color = alt.Color('ColVal:N', scale=scale)
    )

    trdf = df_preds_train.copy()
    trdf.insert(0, 'ColVal', 'Predicted Training Price')

    train_preds_line = alt.Chart(trdf).mark_line().encode(
        x='date:T',
        y='train_pred:Q',
        strokeDash=alt.value([2, 2]),
        color = alt.Color('ColVal:N', scale=scale)  
    )

    tsdf = df_preds_test.copy()
    tsdf.insert(0, 'ColVal', 'Predicted Test Price')

    test_preds_line = alt.Chart(tsdf).mark_line().encode(
        x='date:T',
        y='test_pred:Q',
        strokeDash=alt.value([2, 2]),
        color = alt.Color('ColVal:N', scale=scale)  
    )

    rules = alt.Chart(pd.DataFrame({
    'Date': [start_date_of_test],
    'color': ['red']
    })).mark_rule().encode(
    x='Date:T',
    color=alt.Color('color:N', scale=None)
    )

st.write("You can drag and zoom this plot.")
st.altair_chart((actualBTC + rules + train_preds_line + test_preds_line).properties(
    width=600,
    height=400
).interactive())

st.write("Let's see how we did with an MAE score.")

with st.echo():
    assert test_df.iloc[days_to_predict:].index[0] == df_preds_test.iloc[:-days_to_predict]['date'][0], "Start dates if TEST not equal"
    assert test_df.iloc[days_to_predict:].index[-1] == df_preds_test.iloc[:-days_to_predict]['date'].iloc[-1], "End dates if TEST not equal"

    y_true = test_df.iloc[days_to_predict:]['PriceUSD'] # 7/20 - 9/03
    y_pred = df_preds_test['test_pred'].iloc[:-days_to_predict]

    mae_pred = mean_absolute_error(y_true, y_pred)

st.write("We got an MAE score of: " + str(np.round(mae_pred, 2)) + ". This is the average difference from the true value, which will decrease as the training size percentage increases.")

st.write("Lastly, we'll look at a Residual vs Fit plot to see how well the regressor wctually worked. Ideally you'd see a random distribution round zero here. This is a way to check a suspiciously high R^2 score. If we increase the percentage for the size of size of the training set, the score and plot gets much better.")

plt.clf()
with st.echo():
    plt.scatter(y_true.to_numpy() - y_pred,y_pred)
    plt.title('Residuals versus Fits')

st.pyplot(plt)


print(f'R^2 Score: {np.round(regr.score(X_train,y_train),4)}')


# https://statisticsbyjim.com/regression/interpret-r-squared-regression/

text = '''The data in the fitted line plot follow a very low noise relationship, and the R-squared is 98.5%,
 which seems fantastic. However, the regression line consistently under and over-predicts the data along
  the curve, which is bias. The Residuals versus Fits plot emphasizes this unwanted pattern. 
  An unbiased model has residuals that are randomly scattered around zero. Non-random residual patterns
   indicate a bad fit despite a high R2. Always check your residual plots!'''
plt.clf()










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
