import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from Peilun import clean_dataset
df = clean_dataset()

#d = df but no object type
d = df.loc[:, df.dtypes != np.object]
#Although this is a numerical data, this value everything is 0/NaN, thus unable to use
d = d.drop(['declared_monthly_revenue'], axis=1) 

def cor(d):
    print(d.corr())
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = d.corr()
    sb.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

def Display_filtered_corr_values(d):
    d = pd.DataFrame(d, columns = d.columns)
    corr_pairs = d.corr().unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort")
    print("---corr values that are more than 0.3 scores---\n")
    adjusted = sorted_pairs[:len(sorted_pairs)-len([x for x in sorted_pairs if x==1])]
    head = adjusted[:len([x for x in adjusted if x<-0.3])]
    tail = adjusted[len(adjusted)-len([x for x in adjusted if x>0.3]):]
    result = pd.concat([head,tail])
    result.apply(pd.Series)

def Target_corr(d):
    cor = d.corr()
    print("---Target correlation---")
    #Correlation with output variable
    cor_target = abs(cor["price"])

    #Selecting highly correlated features of a cor higher than 0.3
    relevant_features = cor_target[cor_target>0.3]
    print(relevant_features,"\n")

    print("---To justify that the predictors are not very correlated and able to be used---")
    #Will need to manual change the predicators
    print(df[["payment_value","freight_value"]].corr())
    print(df[["product_weight_g","payment_value"]].corr())


def Model_based_feature_selection(d):
    from sklearn.tree import DecisionTreeClassifier
    kepler_X = d.iloc[:, 1:]
    kepler_y = d.iloc[:, 0]
    clf = DecisionTreeClassifier()
    clf.fit(kepler_X, kepler_y)
    print("---Model-based feature selection (SelectFromModel)---")
    pd.Series(clf.feature_importances_, index=d.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))

def Principal_components(d):
    print("---Principal components---")
    from sklearn.preprocessing import StandardScaler
    cor_mat2 = np.corrcoef(d.T)
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
        
    print("Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. So, the idea is 10-dimensional data gives you 18 principal components, but PCA tries to put maximum possible information in the first component, then maximum remaining information in the second and so on, until having something like shown in the scree plot below.")

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