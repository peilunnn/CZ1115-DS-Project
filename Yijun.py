import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Peilun import clean_dataset
df = clean_dataset()

df_numeric = df.loc[:, df.dtypes != np.object]
#Although declared_monthly_revenue is a numerical data, this value everything is 0/NaN, thus unable to use
df_numeric = df.loc[:, df.dtypes != np.object]
df_numeric = df_numeric.drop(['declared_monthly_revenue'], axis=1) 

'''==================================================================================================PCA=================================================================================================='''

'''Using Pearson Correlation to show all the correlation relationship between one and another.'''  
def Pearson_Corr_values():
    print(df_numeric.corr())
    plt.figure(figsize=(12,10))
    PV_array = df_numeric.corr()
    sb.heatmap(PV_array, annot=True, cmap=plt.cm.Reds)
    plt.show()
    return

'''Display the top and bottom 0.3 Correlation data'''
def Display_filtered_PCV(value):
    corr_pairs = df_numeric.corr().unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    print("---corr values that are more than", value ,"scores---\n")
    adjusted = sorted_pairs[:len(sorted_pairs)-len([x for x in sorted_pairs if x==1])]
    head = adjusted[:len([x for x in adjusted if x<-value])]
    tail = adjusted[len(adjusted)-len([x for x in adjusted if x>value]):]
    result = pd.concat([head,tail])
    print(result.apply(pd.Series))
    return

'''Targets single data and prints out every related correlation relationships, with a value of more than abs(0.3)'''
def Target_corr(value,data):
    PV_array = df_numeric.corr()
    print("---Targeted correlation---")
    #CORRELATION WITH OUTPUT VARIABLE
    cor_target = abs(PV_array[data])

    #SELECTING HIGHLY CORRELATED FEATURES OF A COR HIGHER THAN 0.3
    relevant_features = cor_target[cor_target>value]
    relevant_features = relevant_features[relevant_features<1]
    print(relevant_features,"\n")
    return

'''Displays the features importance in regards to the data using DecisionTreeCrassifier'''
def Model_based_fs():
    kepler_X = df_numeric.iloc[:, 1:]
    kepler_y = df_numeric.iloc[:, 0]
    clf = DecisionTreeClassifier()
    clf.fit(kepler_X, kepler_y)
    print("---Model-based feature selection (SelectFromModel)---")
    pd.Series(clf.feature_importances_, index=df_numeric.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))
    plt.tight_layout() 
    plt.show()
    return

'''Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. 
So, the idea is 10-dimensional data gives you 18 principal components, but PCA tries to put maximum possible information in the first component, then maximum remaining information in the second and so on, until having something like shown in the scree plot below.'''
def Principal_comp():
    print("---Principal components---")
    cor_mat2 = np.corrcoef(df_numeric.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        plt.bar(range(19), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(19), cum_var_exp, where='mid',
                label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout() 
        plt.show()
    return

#----------------------------------------------------------------------------------------------------CONCLUSION----------------------------------------------------------------------------------------------------
#From these we can infer that [payment_value], [product_weight_g] and [geolocation_lng] are the best predictors to predict [price]
#[payment_value] is the best predictor as it has the highest correlation in regards to [price],followed by [product_weight_g] as it has the highest appearance in the top 10 correlation data, while [geolocation_lng] has the highest inverse-correlation and appearance at the negative trend.


'''==================================================================================================REGRESSION=================================================================================================='''

def Predict_Price_Linreg():
    y = pd.DataFrame(df['price'])
    X = pd.DataFrame(df[['payment_value','freight_value','product_weight_g']])  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    print("Train Set :", X_train.shape, y_train.shape)
    print("Test Set  :", X_test.shape, y_test.shape)

    # CREATE A LINEAR REGRESSION OBJECT
    linreg = LinearRegression()

    # TRAIN THE LINEAR REGRESSION MODEL
    linreg.fit(X_train, y_train)

    # PREDICT PRICE VALUES CORRESPONDING TO PREDICTOR
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    # PLOT THE PREDICTIONS VS THE TRUE VALUES
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    axes[0].scatter(y_train, y_train_pred, color = "blue")
    axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
    axes[0].set_xlabel("True values of the Response Variable (Train)")
    axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
    axes[1].scatter(y_test, y_test_pred, color = "green")
    axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
    axes[1].set_xlabel("True values of the Response Variable (Test)")
    axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    plt.show()

    X_pred = pd.DataFrame(df[['payment_value','freight_value','product_weight_g']])  
    y_pred = linreg.predict(X_pred)

    y = pd.DataFrame(df['price'])

    # SUMMARIZE THE ACTUALS, PREDICTIONS AND ERRORS
    y_pred = (pd.DataFrame(y_pred, columns = ["Pred_Price"], index = df.index)).round({'Pred_Price':2})
    Result = pd.concat([X_pred, y_pred], axis = 1)

    y_errs = df["price"]   - Result["Pred_Price"]
    y_errs = pd.DataFrame(y_errs, columns = ["Difference"], index = Result.index)
    Result = pd.concat([Result,df["price"], y_errs], axis = 1)

    df_errs = 100 * abs(Result["price"] - Result["Pred_Price"]) / Result["price"]
    df_errs = pd.DataFrame(df_errs, columns = ["%_Error"], index = Result.index)
    Result = pd.concat([Result,df_errs], axis = 1)

    Result2 = Result.rename(columns={"payment_value":"Payment","freight_value":"Freight","product_weight_g": "PWeight_g"})
    print(Result2.drop_duplicates().head(100))

    # CHECK THE GOODNESS OF FIT (ON TRAIN DATA)
    print("\nGoodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print()

    # CHECK THE GOODNESS OF FIT (ON TEST DATA)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))

def Predict_Price_clf():
    y = pd.DataFrame(df['price'])
    X = pd.DataFrame(df[['payment_value','freight_value','product_weight_g']])  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    print("Train Set :", X_train.shape, y_train.shape)
    print("Test Set  :", X_test.shape, y_test.shape)

    clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
        learning_rate = 0.1, loss='ls')
    clf.fit(X_train, y_train)

    # PREDICT PRICE VALUES CORRESPONDING TO PREDICTORS
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # PLOT THE PREDICTIONS VS THE TRUE VALUES
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    axes[0].scatter(y_train, y_train_pred, color = "blue")
    axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
    axes[0].set_xlabel("True values of the Response Variable (Train)")
    axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
    axes[1].scatter(y_test, y_test_pred, color = "green")
    axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
    axes[1].set_xlabel("True values of the Response Variable (Test)")
    axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    plt.show()

    X_pred = pd.DataFrame(df[['payment_value','freight_value','product_weight_g']])  
    y_pred = clf.predict(X_pred)

    y = pd.DataFrame(df['price'])

    # SUMMARIZE THE ACTUALS, PREDICTIONS AND ERRORS
    y_pred = (pd.DataFrame(y_pred, columns = ["Pred_Price"], index = df.index)).round({'Pred_Price':2})
    Result = pd.concat([X_pred, y_pred], axis = 1)

    y_errs = df["price"]   - Result["Pred_Price"]
    y_errs = pd.DataFrame(y_errs, columns = ["Difference"], index = Result.index)
    Result = pd.concat([Result,df["price"], y_errs], axis = 1)

    df_errs = 100 * abs(Result["price"] - Result["Pred_Price"]) / Result["price"]
    df_errs = pd.DataFrame(df_errs, columns = ["%_Error"], index = Result.index)
    Result = pd.concat([Result,df_errs], axis = 1)

    Result2 = Result.rename(columns={"payment_value":"Payment","freight_value":"Freight","product_weight_g": "PWeight_g"})
    print(Result2.drop_duplicates().head(100))

    # CHECK THE GOODNESS OF FIT (ON TRAIN DATA)
    print("\nGoodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", clf.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print()

    # CHECK THE GOODNESS OF FIT (ON TEST DATA)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", clf.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))



#----------------------------------------------------------------------------------------------------CONCLUSION----------------------------------------------------------------------------------------------------

#1) We import our dependencies, for linear regression we use sklearn (built in python library) and import linear regression from it.
#2) We then initialize Linear Regression to a variable linreg.
#3) We again import another dependency to split our data into train and test.
#4) We've made our train data as 80% and 20% of the data to be our test data , and randomized the splitting of data by using random_state.
#5) Next, we fit our train and test data into linear regression model.
#6) After fitting our data to the model we found out our prediction, aka score of our data, is "49.603" % accurate

#To improve our prediction to atleast 85% target, we use a different method named gradient boosting regression for building a better prediction model as it is also used by many experts, so what is gradient boosting? 
#It is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

#1. We create a variable where we define our gradient boosting regressor and set parameters to it , here:
#	n_estimator — The number of boosting stages to perform. We should not set it too high which would overfit our model.
#	max_depth — The depth of the tree node.
#	learning_rate — Rate of learning the data.
#	loss — loss function to be optimized. ‘ls’ refers to least squares regression
#	minimum sample split — Number of sample to be split for learning the data
#2. We then fit our training data into the gradient boosting model and check for accuracy
#3. We got an accuracy of higher than the previous value!

def main():
    Display_filtered_PCV(0.3)
    Target_corr(0.3,'price')
    Model_based_fs()
    Principal_comp()
    
    Predict_Price_Linreg()
    Predict_Price_clf()

if __name__ == "__main__":
    main()

# sources:                         
# https://likegeeks.com/python-correlation-matrix/  
# https://machinelearningmastery.com/feature-selection-machine-learning-python/  
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b  
# https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn  
# https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html  
# https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
# https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

# https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f