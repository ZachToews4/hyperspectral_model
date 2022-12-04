import numpy as np                                          # Arrays
import os                                                   # OS functions
import pandas as pd                                         # Dataframes
from sklearn.model_selection import train_test_split        # train and test split
from sklearn.ensemble import RandomForestRegressor          # Random Forest
from sklearn.model_selection import RandomizedSearchCV      # Randomized Search

element_temp = 'Ca'     # Change this to the element you want to predict

rf = RandomForestRegressor(random_state = 42)       # Create the model with 100 trees

file_path = "input_data/cleaned_hsi_and_xrf.csv"    # Load training data csv to a dataframe
file = os.path.join(os.getcwd(), file_path)
df = pd.read_csv(file)

X = df                                              # Create an X dataframe
Y = df                                              # Create a Y dataframe

for col in X.columns:
    if col[:4] != 'LWIR' and col[:4] != 'SWIR':
        X = X.drop(columns=col)

for col in Y.columns:
    if col[:4] == 'LWIR' or col[:4] == 'SWIR':
        Y = Y.drop(columns=col)

Y = Y.drop(columns='Sample Location Quality')

for col in X.columns:
    if col[:4] != 'LWIR' and col[:4] != 'SWIR' and col != 'Depth':
        X = X.drop(columns=col)

#Train-Test split
x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
print('Number of training data = ' + str(len(x_train)) + ' and number of testing data = ' + str(len(x_test)))
print('Total number of data = ' + str(len(df.iloc[:,[0]])))

#x_train.sort_values(by=['Depth'], inplace=True)
#Y_train.sort_values(by=['Depth'], inplace=True)
#x_test.sort_values(by=['Depth'], inplace=True)
#Y_test.sort_values(by=['Depth'], inplace=True)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, Y_train[element_temp])

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(x_train, Y_train[element_temp])
base_accuracy = evaluate(base_model, x_test, Y_test[element_temp])

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, Y_test[element_temp])


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

rf_random.best_params_

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
#Al
param_grid = {
    'bootstrap': [False],
    'max_depth': [8,10,12],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [1500, 2000, 2500]
}
#Ca
'''
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 500, 1000, 1600, 2000]
}
'''
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_train, Y_train[element_temp])
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, x_test, Y_test[element_temp])

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
grid_search.best_params_



'''
estimators = 90
while estimators<111:
    for element_temp in ['Al', 'Ca', 'Si']:
        i = 0
        r2 = 0
        while i < 25:
            x_train, x_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
            rfr = RandomForestRegressor(n_estimators=100).fit(x_train, Y_train[element_temp])
            r2 = r2 + r2_score(Y_test[element_temp], rfr.predict(x_test))
            i = i + 1
        print(element_temp, estimators, r2/i)
    estimators = estimators + 1
'''