import pandas as pd
from pandas_profiling import ProfileReport
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
DEBUG = True


def get_dataset(output_path: str = "datasets/df_merged_pickle.pkl") -> None:
    """
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
    print(df.head())
    return df


def EDA(df: pd.DataFrame, output_path: str = "EDA/profile_report.html") -> None:
    """
    """
    prof = ProfileReport(df)
    prof.to_file(output_path)


def main():
    df = clean_dataset()
    # EDA(df)


if __name__ == "__main__":
    main()
