def time_series():
    import pandas as pd
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    from datetime import datetime, date
    from Peilun import clean_dataset
    df = clean_dataset()
    df = df.merge(pd.read_csv("datasets/product_category_name_translation.csv"),
                  how="inner", on="product_category_name")

    # filter date 2017-2018
    df_2017 = df.order_purchase_timestamp.str.contains("2017")
    df = df[df_2017]

    # sort date by order_purchase_timestamp
    df.sort_values(by=['order_purchase_timestamp'], inplace=True)

    # change order_purchase_timestamp to date formate
    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp'])  # change object to datetime
    df_2017 = df[['order_purchase_timestamp',
                  'product_category_name_english', 'product_id']]
    df_2017 = df_2017.rename(
        columns={'product_category_name_english': 'category'})

    # extract product id, month
    df_2017['month'] = df_2017['order_purchase_timestamp'].dt.month

    # filter out top 5 category
    bed = df_2017.category.str.contains("bed_bath_table")
    spo = df_2017.category.str.contains("sports_leisure")
    hea = df_2017.category.str.contains("health_beauty")
    fur = df_2017.category.str.contains("furniture_decor")
    com = df_2017.category.str.contains("computers_accessories")
    bed = df_2017[bed]
    spo = df_2017[spo]
    hea = df_2017[hea]
    fur = df_2017[fur]
    com = df_2017[com]

    # change order_purchase_timestamp to date formate
    bed = bed.replace(['bed_bath_table'], 1)
    bed['category'] = pd.to_numeric(bed['category'])
    spo = spo.replace(['sports_leisure'], 1)
    spo['category'] = pd.to_numeric(spo['category'])
    hea = hea.replace(['health_beauty'], 1)
    hea['category'] = pd.to_numeric(hea['category'])
    fur = fur.replace(['furniture_decor'], 1)
    fur['categoryh'] = pd.to_numeric(fur['category'])
    com = com.replace(['computers_accessories'], 1)
    com['category'] = pd.to_numeric(com['category'])

    # no. of items per month
    bed_month = bed.month.value_counts().sort_index(ascending=True)
    spo_month = spo.month.value_counts().sort_index(ascending=True)
    hea_month = hea.month.value_counts().sort_index(ascending=True)
    fur_month = fur.month.value_counts().sort_index(ascending=True)
    com_month = com.month.value_counts().sort_index(ascending=True)

    # plot the graph
    plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(bed_month, label='bed_bath_table')
    plt.plot(spo_month, label='sports_leisure')
    plt.plot(hea_month, label='health_beauty')
    plt.plot(fur_month, label='furniture_decor')
    plt.plot(com_month, label='computers_accessories')
    plt.legend()


def findings():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from Peilun import clean_dataset
    import folium
    df = clean_dataset()
    df = df.merge(pd.read_csv("datasets/product_category_name_translation.csv"),
                  how="inner", on="product_category_name")
    customer_by_state = df[['customer_unique_id', 'customer_state']].groupby(
        'customer_state').count().reset_index()
    customer_by_state = customer_by_state.sort_values(
        by=['customer_unique_id'])

    # no. of orders per country
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 10))
    plt.bar(customer_by_state['customer_state'],
            customer_by_state['customer_unique_id'])
    plt.show()
    customer_by_state['lat'] = df['geolocation_lat']
    customer_by_state['lon'] = df['geolocation_lng']
    print(df["customer_state"].value_counts())
    #  map
    m = folium.Map(location=[-22.743975, -46.898709],
                   tiles="OpenStreetMap", zoom_start=10)
    for i in range(0, len(customer_by_state)):
        folium.Circle(
            location=[customer_by_state.iloc[i]['lat'],
                      customer_by_state.iloc[i]['lon']],
            popup=customer_by_state.iloc[i]['customer_state'],
            radius=float(
                customer_by_state.iloc[i]['customer_unique_id'])*20,
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(m)
    df = clean_dataset()
    df = df.merge(pd.read_csv("datasets/product_category_name_translation.csv"),
                  how="inner", on="product_category_name")
    customer_by_state = df[['customer_unique_id', 'customer_state']].groupby(
        'customer_state').count().reset_index()
    customer_by_state = customer_by_state.sort_values(
        by=['customer_unique_id'])

    # no. of orders per country
    customer_by_state['lat'] = df['geolocation_lat']
    customer_by_state['lon'] = df['geolocation_lng']

    customer_by_state
    #  map
    m = folium.Map(location=[-22.743975, -46.898709],
                   tiles="OpenStreetMap", zoom_start=5)
    for i in range(0, len(customer_by_state)):
        folium.Circle(
            location=[customer_by_state.iloc[i]['lat'],
                      customer_by_state.iloc[i]['lon']],
            popup=customer_by_state.iloc[i]['customer_state'],
            radius=float(customer_by_state.iloc[i]['customer_unique_id'])*10,
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(m)
    display(m)

    # best selling category
    df['purchase_month'] = pd.DatetimeIndex(
        df['order_purchase_timestamp']).month
    sales_df = df.groupby(['product_category_name'])['price'].sum()
    best_sellers = sales_df.nlargest(10).index

    best_df = df[df['product_category_name'].isin(best_sellers)]

    best_monthly = best_df.pivot_table(
        index='purchase_month', columns='product_category_name_english', values='price', aggfunc='sum')
    best_monthly.plot(kind='bar', figsize=(12, 10))
    plt.title('best selling categories\' monthly earnings')
    plt.xlabel('month')
    plt.ylabel('total earnings')

    # no. of items per month
    # df['order_purchase_year'] = pd.to_datetime(
    #     df['order_purchase_timestamp']).dt.year
    # df['order_purchase_month'] = pd.to_datetime(
    #     df['order_purchase_timestamp']).dt.month
    # orders = df[['order_id', 'order_purchase_year', 'order_purchase_month']]
    # orders = orders.groupby(
    #     ['order_purchase_month', 'order_purchase_year']).count().reset_index()
    # orders = orders.sort_values(
    #     by=['order_purchase_year', 'order_purchase_month'])
    # orders["period"] = orders["order_purchase_month"].astype(
    #     str) + "/" + orders["order_purchase_year"].astype(str)
    # plt.figure(figsize=(15, 10))
    # plt.bar(orders['period'], orders['order_id'])
    # plt.xticks(rotation=75, fontsize=15, weight='bold')
    # plt.yticks(fontsize=15, weight='bold')
    # orders = orders.groupby(
    #     ['order_purchase_month', 'order_purchase_year']).count().reset_index()
    # orders = orders.sort_values(
    #     by=['order_purchase_year', 'order_purchase_month'])
    # orders["period"] = orders["order_purchase_month"].astype(
    #     str) + "/" + orders["order_purchase_year"].astype(str)
    # plt.figure(figsize=(20, 10))
    # my_range = range(1, len(orders.index)+1)
    # plt.stem(orders['order_id'])
    # plt.xticks(my_range, orders['period'])
    #plt.show()




def main():
    time_series()
    findings()


if __name__ == "__main__":
    main()