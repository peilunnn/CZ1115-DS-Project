import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# import plotly.plotly as py
# import plotly.offline as pyoff
# import plotly.graph_objs as go
# from fbprophet.plot import plot_plotly
# import plotly.offline as py
# import plotly.graph_objs as go
from typing import List


# INITIATE PLOTLY
pyoff.init_notebook_mode()
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
DEBUG = True


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
    df_merged.to_pickle(output_path)


def clean_dataset(path: str = f"datasets/{'df_test' if DEBUG else 'df_merged_pickle'}.pkl") -> pd.DataFrame:
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
    # print(CLV_df.head())

    # CONVERTING ORDER PURCHASE TIMESTAMP TO DATETIME OBJECTS
    CLV_df["order_purchase_timestamp"] = pd.to_datetime(
        CLV_df["order_purchase_timestamp"])
    # print(CLV_df.dtypes)

    # EXPLORING THE DATAFRAME
    max_date = CLV_df["order_purchase_timestamp"].dt.date.max()
    min_date = CLV_df["order_purchase_timestamp"].dt.date.min()
    unique_customers = CLV_df["customer_unique_id"].nunique()
    total_sales = CLV_df["price"].sum()

    print(f"Time Range: {min_date} TO {max_date}")
    print(f"No. of unique customers: {unique_customers}")
    print(f"Total sales: {total_sales}")

    return CLV_df


# RFM stands for recency - frequency - Monetary Value. We will group the customers as follows:
# Low Value: Customers who are less active than others, not very frequent buyers and generate very low/zero/negative revenue.
# Mid Value: Customers who often use Olist (but not as much as our High Values) and generates moderate revenue.
# High Value: The group we don’t want to lose. High revenue, frequency and low inactivity.


def CLV_recency(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the recency of each customer, plots the recency scores, shows recency clusters, and updates the CLV dataframe with the recency cluster classifications.
    """

    # We first need to find out the most recent purchase date of each customer and see how many days they are inactive for. After having no. of inactive days for each customer, we will apply k-means clustering to assign customers a recency score.

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
    # Mean is higher than median (243 vs 227), so there is a positive right skew ie. most of the recency values are clustered around the left tail of the distribution.

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

    # We first need to find total number orders for each customer.

    # CREATE AN ORDER COUNTS DATAFRAME
    customer_frequency = CLV_df.groupby(
        "customer_unique_id").order_purchase_timestamp.count().reset_index()
    customer_frequency.columns = ["customer_unique_id", "frequency"]

    # MERGE THE ORDER COUNTS AND CLV DATAFRAMES ON CUSTOMER ID
    CLV_df = pd.merge(CLV_df, customer_frequency,
                      on="customer_unique_id")

    print(CLV_df["frequency"].describe())
    # Mean is higher than median (1.62 vs 1.00), so there is a positive right skew ie. most of the frequency values are clustered around the left tail of the distribution.

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

    # We first need to find revenue for each customer.

    # CREATE A REVENUE DATAFRAME
    customer_revenue = CLV_df.groupby(
        "customer_unique_id").payment_value.sum().reset_index()
    customer_revenue.columns = ["customer_unique_id", "revenue"]

    # MERGE THE CUSTOMER REVENUE AND CLV DATAFRAMES ON CUSTOMER ID
    CLV_df = pd.merge(CLV_df, customer_revenue,
                      on="customer_unique_id")

    print(CLV_df["revenue"].describe())
    # Mean is higher than median (408 vs 137), so there is a positive right skew ie. most of the revenue values are clustered around the left tail of the distribution.

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
    Adds the R/F/M_cluster dataframes to original CLV_df dataframe.
    """

    # ADD 4 CLUSTERS FOR R/F/M TO CLV_df DATAFRAME
    kmeans = KMeans(n_clusters=cluster_no)
    kmeans.fit(CLV_df[[RFM_name]])
    RFM_cluster_name = f"{RFM_name}_cluster"
    CLV_df[RFM_cluster_name] = kmeans.predict(CLV_df[[RFM_name]])
    CLV_df = order_cluster(
        RFM_cluster_name, RFM_name, CLV_df, True)
    print(CLV_df.groupby(RFM_cluster_name)[RFM_name].describe())
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
    # Score 0 is customers with the least value and 8 is customers with the highest value.
    return CLV_df


def categorize(CLV_df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes customers into low/mid/high value based on the overall score.
    """

    # 0 to 2: low value
    # 3 to 4: mid value
    # 5 to 6: high value

    CLV_df["category"] = "low_value"
    CLV_df.loc[CLV_df["overall_score"] > 2, 'category'] = 'mid_value'
    CLV_df.loc[CLV_df['overall_score'] > 4, 'category'] = 'high_value'
    # print(CLV_df.category.sample(10))
    # print(CLV_df.columns)
    return CLV_df


def plot_clusters(CLV_df: pd.DataFrame) -> None:
    """
    Plots each cluster (low, mid or high value) of each pair (recency, frequency, revenue). 
    """

    # REVENUE AGAINST FREQUENCY
    tx_graph = CLV_df.query("revenue < 125000 and frequency < 600")
    x1 = tx_graph.query("category == 'low_value'")['frequency']
    y1 = tx_graph.query("category == 'low_value'")['revenue']

    x2 = tx_graph.query("category == 'mid_value'")['frequency']
    y2 = tx_graph.query("category == 'mid_value'")['revenue']

    x3 = tx_graph.query("category == 'high_value'")['frequency'],
    y3 = tx_graph.query("category == 'high_value'")['revenue'],

    # ax = CLV_df.plot(kind='scatter', x=x1, y=y1, color='DarkBlue', label='Group 1')
    # CLV_df.plot(kind='scatter', x=x2, y=y2, color='DarkGreen', label='Group 2', ax=ax)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x1, y1, s=10, c='b', marker="s", label='low')
    ax1.scatter(x2, y2, s=10, c='r', marker="o", label='mid')
    ax1.scatter(x3, y3, s=10, c='g', marker="o", label='high')
    plt.legend(loc='upper left')


def train_test(CLV_df: pd.DataFrame):
    """
    """
    # # WE TAKE 3 MONTHS OF DATA, CALCULATE RFM AND USE IT TO PREDICT CLV FOR THE NEXT 6 MONTHS
    # print(CLV_df.order_purchase_timestamp.min())
    # print(CLV_df.order_purchase_timestamp.max())
    # CLV_3m = CLV_df.loc['2016-10-04':'2016-02-01']

    # CLV_df[(CLV_df.order_purchase_timestamp < date(2016, 10, 4)) & (
    #     CLV_df.order_purchase_timestamp >= date(2016, 10, 4))].reset_index(drop=True)

    # tx_6m = CLV_df[(CLV_df.order_purchase_timestamp >= date(2011, 6, 1)) & (
    # CLV_df.order_purchase_timestamp < date(2011, 12, 1))].reset_index(drop=True)

    cols_extracted = ["customer_unique_id", "recency", "recency_cluster",
                      "frequency", "frequency_cluster", "revenue", "revenue_cluster", "overall_score", "category"]
    feature_set = CLV_df[cols_extracted]


def main():
    df = clean_dataset()
    # EDA(df)
    CLV_df = CLV_EDA(df)
    CLV_df = CLV_recency(CLV_df)
    CLV_df = CLV_frequency(CLV_df)
    CLV_df = CLV_revenue(CLV_df)
    CLV_df = overall_RFM(CLV_df)
    CLV_df = categorize(CLV_df)
    # print(CLV_df.head())
    plot_clusters(CLV_df)
    # train_test(CLV_df)


if __name__ == "__main__":
    main()
