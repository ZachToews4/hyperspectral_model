# Predicting Elemental Weight Percents from Hyperspectral Core Image Data using XRF and Machine Learning
# 
# 
# By Zach Toews modified
# 
# Department of Geoscience
# 
# The University of Calgary
# 
# **********************************************************************************************************************
# 
# **Approach:**
# 1. XRF and hyperspectral data was cleaned in Excel. 
# 2. Only the hyperspectral data that matches with an XRF data point are included in the training set.
# 3. The training data is used to build and tune a model.
# 4. An XRF prediction is then made on every hyperspectal data point using the model.
# 
# **********************************************************************************************************************


# IMPORT AND DATA LOADING
# import packages
import os                                                   # Operating system
import pandas as pd                                         # Dataframes
from sklearn.linear_model import LinearRegression           # linear regression
from sklearn.metrics import r2_score                        # R2 score
from sklearn.model_selection import train_test_split        # train and test split
from sklearn.ensemble import RandomForestRegressor          # Random Forest
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

# Load training data csv to a dataframe
file_path = "input_data/cleaned_hsi_and_xrf.csv"
file = os.path.join(os.getcwd(), file_path) 
df = pd.read_csv(file)

# Create X and Y dataframes
X = df
Y = df
for col in X.columns:
    if col[:4] != 'LWIR' and col[:4] != 'SWIR':
        X = X.drop(columns=col)

for col in Y.columns:
    if col[:4] == 'LWIR' or col[:4] == 'SWIR':
        Y = Y.drop(columns=col)

Y = Y.drop(columns='Sample Location Quality')


# LINEAR REGRESSION TRAINING
elements_r2 = []
Y_pred = pd.DataFrame()
n = 0
lin_reg = LinearRegression()
for elm in Y.columns:
    lin_reg.fit(X, Y[elm])
    elements_r2.append([])
    elements_r2[n].append(elm)
    Y_pred[elm] = lin_reg.predict(X)
    elements_r2[n].append(r2_score(Y[elm], Y_pred[elm]))
    n = n + 1

elements_r2 = sorted(elements_r2, key=lambda x: x[1], reverse=True)
print('Linear Regression R2 Scores:')
for elm in elements_r2:
    print(elm[0], elm[1])

dfExport = pd.DataFrame()
for elm in Y.columns:
    dfExport[elm] = Y[elm]
    dfExport[elm + '_pred'] = Y_pred[elm]

dfExport.to_csv("LinReg-training.csv")


# LINEAR REGRESSION PREDICTION ONLY
file_path_HSI = "input_data/cleaned_hsi_only.csv"
file_HSI = os.path.join(os.getcwd(), file_path_HSI) 
df_HSI = pd.read_csv(file_HSI)
X_HSI = df_HSI
for col in X_HSI.columns:
    if col[:4] != 'LWIR' and col[:4] != 'SWIR':
        X_HSI = X_HSI.drop(columns=col)

Y_pred_HSI = pd.DataFrame()
for elm in Y.columns:
    #lin_reg = LinearRegression()
    lin_reg.fit(X, Y[elm])
    Y_pred_HSI[elm] = lin_reg.predict(X_HSI)

dfExport = pd.DataFrame()
for elm in Y.columns:
    dfExport[elm + '_pred'] = Y_pred_HSI[elm]

dfExport.to_csv("LinReg-prediction-only.csv")


# TRAIN TEST SPLIT
X = df
for col in X.columns:
    if col[:4] != 'LWIR' and col[:4] != 'SWIR' and col != 'Depth':
        X = X.drop(columns=col)
x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
print('Number of training data = ' + str(len(x_train)))
print('Number of testing data = ' + str(len(x_test)))
print('Total number of data = ' + str(len(df.iloc[:,[0]])))
x_train.sort_values(by=['Depth'], inplace=True)
Y_train.sort_values(by=['Depth'], inplace=True)
x_test.sort_values(by=['Depth'], inplace=True)
Y_test.sort_values(by=['Depth'], inplace=True)
x_test = x_test.drop(columns='Depth')
x_train = x_train.drop(columns='Depth')


# RANDOM FOREST R2 SCORES
element_temp = 'HLD'
i = 0
r2 = 0
while i < 100:
    x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    rfr = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500).fit(x_train, Y_train[element_temp])
    r2 = r2 + r2_score(Y_test[element_temp], rfr.predict(x_test))
    i = i + 1
print(element_temp, r2/i)

element_temp = 'Ca'
i = 0
r2 = 0
while i < 100:
    x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    rfr = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500).fit(x_train, Y_train[element_temp])
    r2 = r2 + r2_score(Y_test[element_temp], rfr.predict(x_test))
    i = i + 1
print(element_temp, r2/i)

element_temp = 'Al'
i = 0
r2 = 0
while i < 100:
    x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    rfr = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500).fit(x_train, Y_train[element_temp])
    r2 = r2 + r2_score(Y_test[element_temp], rfr.predict(x_test))
    i = i + 1
print(element_temp, r2/i)

element_temp = 'Si'
i = 0
r2 = 0
while i < 100:
    x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    rfr = RandomForestRegressor(bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=500).fit(x_train, Y_train[element_temp])
    r2 = r2 + r2_score(Y_test[element_temp], rfr.predict(x_test))
    i = i + 1
print(element_temp, r2/i)
