import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from Peilun import clean_dataset
df = clean_dataset()
#d = df but no object type
df_numeric = df.loc[:, df.dtypes != np.object]
#Although declared_monthly_revenue is a numerical data, this value everything is 0/NaN, thus unable to use
df_numeric = df.loc[:, df.dtypes != np.object]
df_numeric = df_numeric.drop(['declared_monthly_revenue'], axis=1) 


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
    #Correlation with output variable
    cor_target = abs(PV_array[data])

    #Selecting highly correlated features of a cor higher than 0.3
    relevant_features = cor_target[cor_target>value]
    relevant_features = relevant_features[relevant_features<1]
    print(relevant_features,"\n")

    # print("---To justify that the predictors are not very correlated and able to be used---")
    # #Will need to manual change the predicators
    # print(df[["payment_value","freight_value"]].corr())
    # print(df[["product_weight_g","payment_value"]].corr())
    return

'''Displays the features importance in regards to the data using DecisionTreeCrassifier'''
def Model_based_fs():
    kepler_X = df_numeric.iloc[:, 1:]
    kepler_y = df_numeric.iloc[:, 0]
    clf = DecisionTreeClassifier()
    clf.fit(kepler_X, kepler_y)
    print("---Model-based feature selection (SelectFromModel)---")
    pd.Series(clf.feature_importances_, index=df_numeric.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))
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

#From these we can infer that [Price] is the best predictor, followed by [payment_value], [product_weight_g] and [geolocation_lng]
#[Price] is the best predictor followed by [payment_value] as it has the highest correlation in regards to the dataset,and [product_weight_g] is chosen because it has the highest appearance in the top 10 correlation data, while [geolocation_lng] has the highest inverse-correlation and appearance at the negative trend.

def main():
    Display_filtered_PCV(0.3)
    Target_corr(0.3,'price')
    Model_based_fs()
    Principal_comp()

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