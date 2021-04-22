from typing import List
import pandas as pd
import math
from pandas_profiling import ProfileReport
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

DEBUG = False


def get_dataset(output_path: str = "datasets/df_merged_pickle.pkl") -> None:
    """
    Merges each of the csv datasets on a foreign key and returns a pickle file for easy conversion to a dataframe.
    """
    df_merged_1 = pd.read_csv("datasets/olist_order_reviews_dataset.csv").merge(
        pd.read_csv("datasets/olist_orders_dataset.csv"), how="inner", on="order_id").merge(
        pd.read_csv("datasets/olist_order_payments_dataset.csv"), how="inner", on="order_id").merge(
        pd.read_csv("datasets/olist_order_items_dataset.csv"), how="inner", on="order_id").merge(
        pd.read_csv("datasets/olist_products_dataset.csv"), how="inner", on="product_id")

    df_merged_2 = pd.read_csv("datasets/olist_sellers_dataset.csv").merge(
        pd.read_csv("datasets/olist_geolocation_dataset.csv"), how="inner", left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix").merge(
        pd.read_csv("datasets/olist_customers_dataset.csv"), how="inner", left_on="geolocation_zip_code_prefix", right_on="customer_zip_code_prefix").merge(
        pd.read_csv("datasets/olist_closed_deals_dataset.csv"), how="inner", on="seller_id").merge(
        pd.read_csv("datasets/olist_marketing_qualified_leads_dataset.csv"), how="inner", on="mql_id")

    df_merged = df_merged_1.merge(df_merged_2, how="inner", on="customer_id")
    df_merged.to_pickle(output_path, protocol=4)

    df_100k = df_merged.sample(100000)
    df_test_prot.to_pickle("datasets/df_100k.pkl", protocol=4)


def clean_dataset(path: str = f"datasets/{'df_test' if DEBUG else 'df_100k'}.pkl") -> pd.DataFrame:
    """
    Removes duplicate columns, renaming and dropping columns, filling in NaN values and returns the cleaned dataframe.
    """
    df = pd.read_pickle(path)

    # REMOVE DUPLICATE COLUMNS AND RENAME THEM
    df = df[[col for col in df.columns if not col.endswith("_x")]]
    df.columns = [
        col[:-2] if col.endswith("_y") else col for col in df.columns]

    # # FIND PERCENTAGE OF MISSING VALUES IN EACH COLUMN
    # pct_missing = df.isnull().sum() * 100 / len(df)
    # print(pct_missing)

    # DROP COLUMNS WITH MORE THAN 50% NULL VALUES
    df = df.loc[:, df.isnull().mean() < 0.5]

    # FOR COLUMNS WITH LESS THAN 5% MISSING VALUES, JUST DROP THE ROWS WITH THOSE VALUES
    df_5_pct = df.loc[:, (df.isnull().mean() > 0) &
                      (df.isnull().mean() < 0.05)]
    df = df.dropna(how='any', subset=df_5_pct.columns)

    # WE ARE LEFT WITH CATEGORICAL LEAD_BEHAVIOUR_PROFILE COLUMN TO CLEAN UP W 22% MISSING VALUES
    # WE FILL THE MISSING VALUES WITH THE MOST COMMON CATEGORY
    df = df.fillna(df['lead_behaviour_profile'].value_counts().index[0])
    # print(df.head())
    return df


def EDA(df: pd.DataFrame, output_path: str = "EDA/profile_report.html") -> None:
    """
    Generates a pandas profiling report.
    """
    prof = ProfileReport(df)
    prof.to_file(output_path)


def CLV_EDA(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups all customers based on their start month, and using frequency and recency, predicts the class that each customer falls into based on their CLV.
    """

    # Customer Lifetime Value (CLV) is the total monetary value of purchases made by a customer with a business over his entire lifetime - the time period that the customer purchases with the business before moving to your competitors.

    # EXTRACT ONLY THE RELEVANT COLUMNS`
    cols_CLV = ["order_id", "product_id", "order_purchase_timestamp", "payment_value",
                "price", "customer_unique_id", "geolocation_state"]
    CLV_df = df.copy()
    CLV_df = CLV_df[cols_CLV]

    # CONVERTING ORDER PURCHASE TIMESTAMP TO DATETIME OBJECTS
    CLV_df["order_purchase_timestamp"] = pd.to_datetime(
        CLV_df["order_purchase_timestamp"])

    # EXPLORING THE DATAFRAME
    max_date = CLV_df["order_purchase_timestamp"].dt.date.max()
    min_date = CLV_df["order_purchase_timestamp"].dt.date.min()
    unique_customers = CLV_df["customer_unique_id"].nunique()
    total_sales = CLV_df["price"].sum()

    print(f"Time Range: {min_date} TO {max_date}")
    print(f"No. of unique customers: {unique_customers}")
    print(f"Total sales: {total_sales}\n")

    return CLV_df


# ----------------------------------CLUSTERING-----------------------------
def CLV_recency(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the recency of each customer, plots the recency scores, shows recency clusters, and updates the CLV dataframe with the recency cluster classifications.
    """
    # CREATE A CUSTOMER MAX PURCHASE DATAFRAME FOR MAX PURCHASE DATE OF EACH CUSTOMER
    customer_max_purchase = CLV_df.groupby(
        "customer_unique_id").order_purchase_timestamp.max().reset_index()
    customer_max_purchase.columns = ["customer_unique_id", "max_purchase_date"]

    # CREATE A RECENCY COLUMN IN CUSTOMER MAX PURCHASE DATAFRAME
    customer_max_purchase["recency"] = (customer_max_purchase['max_purchase_date'].max(
    ) - customer_max_purchase['max_purchase_date']).dt.days

    # MERGE THE ORIGINAL AND CUSTOMER MAX PURCHASE DATAFRAMES ON CUSTOMER ID
    CLV_df = pd.merge(
        CLV_df, customer_max_purchase[["customer_unique_id", "recency"]], on="customer_unique_id")

    print(CLV_df["recency"].describe())
    print("\n")

    # PLOT A RECENCY HISTOGRAM
    sb.histplot(data=CLV_df["recency"])
    plt.show()

    # FIND OPTIMAL NUMBER OF CLUSTERS, DO THE CLUSTERING AND ADD CLUSTERS TO CLV DATAFRAME
    elbow_method(CLV_df, "recency")
    CLV_df = k_means_clustering(CLV_df, "recency", 4)
    return CLV_df


def CLV_frequency(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the frequency of each customer, plots the frequency scores, shows frequency clusters, and updates the CLV dataframe with the frequency cluster classifications.
    """

    # CREATE AN ORDER COUNTS DATAFRAME
    customer_frequency = CLV_df.groupby(
        "customer_unique_id").order_purchase_timestamp.count().reset_index()
    customer_frequency.columns = ["customer_unique_id", "frequency"]

    # MERGE THE ORDER COUNTS AND CLV DATAFRAMES ON CUSTOMER ID
    CLV_df = pd.merge(CLV_df, customer_frequency,
                      on="customer_unique_id")

    print(CLV_df["frequency"].describe())
    print("\n")

    # PLOT A FREQUENCY HISTOGRAM
    sb.histplot(data=CLV_df["frequency"])
    plt.show()

    # FIND OPTIMAL NUMBER OF CLUSTERS, DO THE CLUSTERING AND ADD CLUSTERS TO CLV DATAFRAME
    elbow_method(CLV_df, "frequency")
    CLV_df = k_means_clustering(CLV_df, "frequency", 4)
    return CLV_df


def CLV_revenue(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the revenue of each customer, plots the revenue scores, shows revenue clusters, and updates the CLV dataframe with the revenue cluster classifications.
    """

    # CREATE A REVENUE DATAFRAME
    customer_revenue = CLV_df.groupby(
        "customer_unique_id").payment_value.sum().reset_index()
    customer_revenue.columns = ["customer_unique_id", "revenue"]

    # MERGE THE CUSTOMER REVENUE AND CLV DATAFRAMES ON CUSTOMER ID
    CLV_df = pd.merge(CLV_df, customer_revenue,
                      on="customer_unique_id")

    print(CLV_df["revenue"].describe())
    print("\n")

    # PLOT A REVENUE HISTOGRAM
    sb.histplot(data=CLV_df["revenue"])
    plt.show()

    # FIND OPTIMAL NUMBER OF CLUSTERS, DO THE CLUSTERING AND ADD CLUSTERS TO CLV DATAFRAME
    elbow_method(CLV_df, "revenue")
    CLV_df = k_means_clustering(CLV_df, "revenue", 3)
    return CLV_df


def elbow_method(CLV_df: pd.DataFrame, RFM_name: str) -> None:
    """
    Finds optimal number of clusters for R/F/M.
    """

    # k-means clustering to assign customers a recency score: we choose k number of clusters and randomly select the centroid for each cluster. We then assign all points to the closest centroid, and recompute the centroids of the new clusters and do this until there is no point that is changing from one cluster to another. But to find the optimal cluster number, we will use elbow method, which plots a graph of inertia, which calculates the sum of distances of all the points within a cluster from the centroid of that cluster, against the number of clusters. The cluster value where the decrease in inertia value becomes constant is the optimal cluster number.

    sse = {}
    RFM_var = CLV_df[[RFM_name]].copy()
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(RFM_var)
        RFM_var["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("No. of clusters")
    plt.show()

    # From the inertia graph, the cluster value where the decrease in inertia value becomes constant is 4.This is the optimal cluster number.


def k_means_clustering(CLV_df: pd.DataFrame, RFM_name: str, cluster_no: int) -> pd.DataFrame:
    """
    Does clustering and adds the ordered R/F/M_cluster dataframes to original CLV_df dataframe.
    """

    # ADD 4 CLUSTERS FOR R/F/M TO CLV_df DATAFRAME
    kmeans = KMeans(n_clusters=cluster_no)
    kmeans.fit(CLV_df[[RFM_name]])
    RFM_cluster_name = f"{RFM_name}_cluster"
    CLV_df[RFM_cluster_name] = kmeans.predict(CLV_df[[RFM_name]])
    CLV_df = order_cluster(
        RFM_cluster_name, RFM_name, CLV_df, True)
    print(CLV_df.groupby(RFM_cluster_name)[RFM_name].describe())
    print("\n")
    return CLV_df


def order_cluster(cluster_name: str, target_name: str, df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """
    Sorts the cluster numbers in ascending order
    """
    df_new = df.groupby(cluster_name)[
        target_name].mean().reset_index()
    df_new = df_new.sort_values(
        by=target_name, ascending=ascending).reset_index(drop=True)
    df_new["index"] = df_new.index
    df_final = pd.merge(
        df, df_new[[cluster_name, 'index']], on=cluster_name)
    df_final = df_final.drop([cluster_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_name})
    return df_final


def overall_RFM(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses all 3 RFM scores to calculate an overall score to rank each customer.
    """
    CLV_df["overall_score"] = CLV_df["recency_cluster"] + \
        CLV_df["frequency_cluster"] + CLV_df["revenue_cluster"]
    print(CLV_df.groupby("overall_score")[[
          "recency", "frequency", "revenue"]].mean())
    return CLV_df


def categorize(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes customers into low/mid/high value based on the overall score.
    """

    CLV_df["category"] = "low_value"
    CLV_df.loc[CLV_df["overall_score"] > 1, 'category'] = 'mid_value'
    CLV_df.loc[CLV_df['overall_score'] > 3, 'category'] = 'high_value'
    return CLV_df


def plot_clusters(CLV_df: pd.DataFrame) -> None:
    """
    Plots each cluster (low, mid or high value) of each pair (recency, frequency, revenue).
    """

    # REVENUE AGAINST FREQUENCY
    graph1 = CLV_df.query("revenue < 4000000 and frequency < 7000")
    x1_low = graph1.query("category == 'low_value'")['frequency']
    y1_low = graph1.query("category == 'low_value'")['revenue']

    x1_mid = graph1.query("category == 'mid_value'")['frequency']
    y1_mid = graph1.query("category == 'mid_value'")['revenue']

    x1_high = graph1.query("category == 'high_value'")['frequency'],
    y1_high = graph1.query("category == 'high_value'")['revenue'],

    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot(111)
    ax.scatter(x1_low, y1_low, s=10, c='b', marker="s", label='low')
    ax.scatter(x1_mid, y1_mid, s=10, c='r', marker="o", label='mid')
    ax.scatter(x1_high, y1_high, s=10, c='g', marker="o", label='high')
    plt.legend(loc='upper right')
    plt.title("Revenue against Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Revenue")

    # REVENUE AGAINST RECENCY
    graph2 = CLV_df.query("revenue < 4000000 and recency < 700")
    x2_low = graph2.query("category == 'low_value'")['recency']
    y2_low = graph2.query("category == 'low_value'")['revenue']

    x2_mid = graph2.query("category == 'mid_value'")['recency']
    y2_mid = graph2.query("category == 'mid_value'")['revenue']

    x2_high = graph2.query("category == 'high_value'")['recency'],
    y2_high = graph2.query("category == 'high_value'")['revenue'],

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.scatter(x2_low, y2_low, s=10, c='b', marker="s", label='low')
    ax.scatter(x2_mid, y2_mid, s=10, c='r', marker="o", label='mid')
    ax.scatter(x2_high, y2_high, s=10, c='g', marker="o", label='high')
    plt.legend(loc='upper right')
    plt.title("Revenue against Recency")
    plt.xlabel("Recency")
    plt.ylabel("Revenue")

    # RECENCY AGAINST FREQUENCY
    graph3 = CLV_df.query("frequency < 7000 and recency < 700")
    x3_low = graph3.query("category == 'low_value'")['frequency']
    y3_low = graph3.query("category == 'low_value'")['recency']

    x3_mid = graph3.query("category == 'mid_value'")['frequency']
    y3_mid = graph3.query("category == 'mid_value'")['recency']

    x3_high = graph3.query("category == 'high_value'")['frequency'],
    y3_high = graph3.query("category == 'high_value'")['recency'],

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.scatter(x3_low, y3_low, s=10, c='b', marker="s", label='low')
    ax.scatter(x3_mid, y3_mid, s=10, c='r', marker="o", label='mid')
    ax.scatter(x3_high, y3_high, s=10, c='g', marker="o", label='high')
    plt.legend(loc='upper right')
    plt.title("Frequency against Recency")
    plt.xlabel("Recency")
    plt.ylabel("Frequency")


# -------------------------CLASSIFICATION------------------
def split_months(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the 21 months of data in CLV_df into 7 months and 14 months for train test set later.
    """
    CLV_df_copy = CLV_df.copy()
    CLV_df_copy = CLV_df_copy.set_index(["order_purchase_timestamp"])
    CLV_7m = CLV_df_copy.loc['2016-10-01':'2017-06-01']
    CLV_7m = CLV_7m[["customer_unique_id", "recency", "recency_cluster", "frequency",
                     "frequency_cluster", "revenue", "revenue_cluster", "overall_score", "category"]]
    CLV_14m = CLV_df_copy.loc['2016-07-01':'2018-08-01']
    CLV_14m = CLV_14m[["customer_unique_id", "recency", "recency_cluster", "frequency",
                       "frequency_cluster", "revenue", "revenue_cluster", "overall_score", "category"]]

    CLV_14m["CLV"] = CLV_14m["revenue"]

    return CLV_7m, CLV_14m


def get_CLV_merged(CLV_14m: pd.DataFrame, CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the original dataframe CLV_df with 14 months dataframe CLV_df_14m and returns the merged dataframe CLV_merged.
    """
    CLV_df_14m = CLV_14m.groupby('customer_unique_id')[
        'revenue'].sum().reset_index()
    CLV_df_14m.columns = ['customer_unique_id', 'm14_Revenue']
    CLV_merged = pd.merge(CLV_df, CLV_df_14m,
                          on='customer_unique_id', how='left')
    CLV_merged = CLV_merged.fillna(0)
    CLV_merged = CLV_merged[CLV_merged['m14_Revenue']
                            < CLV_merged['m14_Revenue'].quantile(0.99)]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(CLV_merged[['m14_Revenue']])
    CLV_merged['CLV_cluster'] = kmeans.predict(CLV_merged[['m14_Revenue']])

    # order cluster number based on LTV
    CLV_merged = order_cluster('CLV_cluster', 'm14_Revenue', CLV_merged, True)

    print(CLV_merged.groupby('CLV_cluster')['m14_Revenue'].describe())
    print("\n")
    return CLV_merged


def correlation_plot(CLV_merged: pd.DataFrame) -> None:
    """
    Plots a scatter plot of 14m CLV against overall_score.
    """
    graph = CLV_merged.query("m14_Revenue < 4000000")
    x_low = graph.query("category == 'low_value'")['overall_score']
    y_low = graph.query("category == 'low_value'")['m14_Revenue']

    x_mid = graph.query("category == 'mid_value'")['overall_score']
    y_mid = graph.query("category == 'mid_value'")['m14_Revenue']

    x_high = graph.query("category == 'high_value'")['overall_score'],
    y_high = graph.query("category == 'high_value'")['m14_Revenue'],

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.scatter(x_low, y_low, s=10, c='b', marker="s", label='low')
    ax.scatter(x_mid, y_mid, s=10, c='r', marker="o", label='mid')
    ax.scatter(x_high, y_high, s=10, c='g', marker="o", label='high')
    plt.legend(loc='upper right')
    plt.title("14m CLV against overall_score")
    plt.xlabel("overall_score")
    plt.ylabel("14m CLV")


def one_hot_encoding(CLV_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Converts values in categorical column "category" in CLV_merged into numeric values 0 and 1.
    """
    return pd.concat([CLV_merged, pd.get_dummies(CLV_merged["category"])], axis=1).rename(columns={"high_value": "category_high_value", "low_value": "category_low_value", "mid_value": "category_mid_value"}).drop(['order_id', 'product_id', 'customer_unique_id', 'geolocation_state', 'category', 'order_purchase_timestamp'], axis=1)


def print_correlation(CLV_merged: pd.DataFrame) -> None:
    """
    Prints the correlation of each of the columns with our response variable CLV_cluster.
    """
    # calculate and show correlations
    corr_matrix = CLV_merged.corr()
    print(corr_matrix['CLV_cluster'].sort_values(ascending=False))


def XGB_classification(CLV_merged: pd.DataFrame) -> None:
    """
    Train our data using XGB to predict a cluster that a customer would fall into. ALso prints the classification report.
    """
    # create X and y, X will be feature set and y is the label - LTV
    X = CLV_merged[['category_low_value',
                    'category_mid_value', 'category_high_value']]
    y = CLV_merged['CLV_cluster']

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=56)

    # XGBoost Multiclassification Model
    ltv_xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.1, objective='multi:softprob', n_jobs=-1).fit(X_train, y_train)

    print('Accuracy of XGB classifier on training set: {:.2f}'
          .format(ltv_xgb_model.score(X_train, y_train)))
    print('Accuracy of XGB classifier on test set: {:.2f}'
          .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

    y_pred = ltv_xgb_model.predict(X_test)
    # print(classification_report(y_test, y_pred))

    matrix = confusion_matrix(y_test, y_pred)
    sb.heatmap(matrix, annot=True, fmt=".0f", annot_kws={"size": 6})
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = ((1 + 2**2) * (precision * recall)) / \
        ((1 + 2 ** 2) * precision + recall)
    print(f"f1 score is: {f1[0]:.3f}")


# def main():
#     # get_dataset()
#     df = clean_dataset()
#     # EDA(df)
#     CLV_df = CLV_EDA(df)
#     CLV_df = CLV_recency(CLV_df)
#     CLV_df = CLV_frequency(CLV_df)
#     CLV_df = CLV_revenue(CLV_df)
#     CLV_df = overall_RFM(CLV_df)
#     CLV_df = categorize(CLV_df)
#     plot_clusters(CLV_df)
#     CLV_7m, CLV_14m = split_months(CLV_df)
#     CLV_merged = get_CLV_merged(CLV_14m, CLV_df)
#     correlation_plot(CLV_merged)
#     print_correlation(CLV_merged)
#     CLV_merged = one_hot_encoding(CLV_merged)
#     XGB_classification(CLV_merged)


# if __name__ == "__main__":
#     main()
