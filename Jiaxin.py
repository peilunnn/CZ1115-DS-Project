def jiaxin():
    import pandas as pd 
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    from datetime import datetime,date
    from Peilun import clean_dataset
    df=clean_dataset()
    df = df.merge(pd.read_csv("datasets/product_category_name_translation.csv"),how="inner", on="product_category_name")
    
    #filter date 2017-2018
    df_2017=df.order_purchase_timestamp.str.contains("2017")
    df=df[df_2017]
    
    #sort date by order_purchase_timestamp
    df.sort_values(by=['order_purchase_timestamp'], inplace=True)
    
    #change order_purchase_timestamp to date formate
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp']) #change object to datetime
    df_2017=df[['order_purchase_timestamp','product_category_name_english','product_id']]
    df_2017= df_2017.rename(columns={'product_category_name_english': 'category'})
    
    #extract product id, month
    df_2017['month']=df_2017['order_purchase_timestamp'].dt.month
    
    #filter out top 5 category
    bed=df_2017.category.str.contains("bed_bath_table")
    spo=df_2017.category.str.contains("sports_leisure")
    hea=df_2017.category.str.contains("health_beauty")
    fur=df_2017.category.str.contains("furniture_decor")
    com=df_2017.category.str.contains("computers_accessories")
    bed=df_2017[bed]
    spo=df_2017[spo]
    hea=df_2017[hea]
    fur=df_2017[fur]
    com=df_2017[com]
    
    #change order_purchase_timestamp to date formate
    bed = bed.replace(['bed_bath_table'],1)
    bed['category'] = pd.to_numeric(bed['category'])
    spo = spo.replace(['sports_leisure'],1)
    spo['category'] = pd.to_numeric(spo['category'])
    hea = hea.replace(['health_beauty'],1)
    hea['category'] = pd.to_numeric(hea['category'])
    fur = fur.replace(['furniture_decor'],1)
    fur['categoryh'] = pd.to_numeric(fur['category'])
    com = com.replace(['computers_accessories'],1)
    com['category'] = pd.to_numeric(com['category'])
    
    #no. of items per month
    bed_month=bed.month.value_counts().sort_index(ascending=True)
    spo_month=spo.month.value_counts().sort_index(ascending=True)
    hea_month=hea.month.value_counts().sort_index(ascending=True)
    fur_month=fur.month.value_counts().sort_index(ascending=True)
    com_month=com.month.value_counts().sort_index(ascending=True)
    
    #plot the graph
    plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    plt.plot(bed_month,label='bed_bath_table')
    plt.plot(spo_month,label='sports_leisure')
    plt.plot(hea_month,label='health_beauty')
    plt.plot(fur_month,label='furniture_decor')
    plt.plot(com_month,label='computers_accessories')
    plt.legend()
