import pandas as pd
from tabulate import tabulate

def preprocessing_dataframe(df):
    # df= pd.read_csv("IMPA_ITEMS_WITH_PORT.csv",encoding='latin-1')
    import numpy as np
    df['RECEIPT_DATE'] = df['RECEIPT_DATE'].apply(lambda x: str(x).split(' ', 1)[0] if ' ' in str(x) else str(x))
    df['RECEIPT_DATE'] = pd.to_datetime(df['RECEIPT_DATE'], format='%d-%m-%Y')
    df['RECEIPT_DATE'] = df['RECEIPT_DATE'].dt.date
    df['APPROVED_DATE'] = pd.to_datetime(df['APPROVED_DATE'], format='%d-%m-%Y')
    df['APPROVED_DATE'] = df['APPROVED_DATE'].dt.date
    df.dropna(subset=['DR_QTY'], inplace=True)
    df.dropna(subset=['AC_QTY'], inplace=True)
    df['ITEM_NO_STOCK']=df['PO_QTY']-df['DR_QTY']
    df['ITEM_QC_NOT_OK']= df['DR_QTY']-df['AC_QTY']
    df['ITEM_SECTION1_ID'].replace('', pd.NaT, inplace=True)
    filtered_df = df.dropna(subset=['PART_NUMBER'])
    filtered_df = filtered_df.dropna(subset=['ITEM'])
    filtered_df = filtered_df.dropna(subset=['PORT_ID'])
    filtered_df.drop(columns=['ITEM_SECTION_3'], inplace=True)
    filtered_df.drop(columns=['REASON_ID'], inplace=True)
    filtered_df.drop(columns=['REASON'], inplace=True)
    filtered_df = filtered_df.dropna(subset=['ITEM_SECTION1_ID'])
    filtered_df.reset_index(drop=True,inplace=True)
    filtered_df['PART_NUMBER_PROCESSED'] = filtered_df['PART_NUMBER'].astype(str)
    filtered_df['PART_NUMBER_PROCESSED'] = filtered_df['PART_NUMBER_PROCESSED'].str.lower()
    filtered_df['ITEM_PROCESSED'] = filtered_df['ITEM'].astype(str)
    filtered_df['ITEM_PROCESSED'] = filtered_df['ITEM_PROCESSED'].str.lower()
    filtered_df['ITEM_PROCESSED'] = filtered_df['ITEM_PROCESSED'].apply(clean_item_name)
    outliers = filtered_df.groupby(['VENDOR', 'ITEM_PROCESSED','PO_APPROVED_YEAR'])['UNIT_PRICE'].apply(
    lambda x: x[(x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
                (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))])
    outliers = outliers.reset_index(level=[0, 1])
    outliers = pd.merge(outliers, filtered_df[['VENDOR', 'ITEM_PROCESSED', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']], how='left',
                        on=['VENDOR', 'ITEM_PROCESSED'])
    outliers_df=outliers.drop_duplicates()
    merged_df = pd.merge(filtered_df, outliers_df, how='left', indicator=True)
    df_without_outliers = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_without_outliers.drop_duplicates(inplace=True)
    df_without_outliers['TOTALAMOUNT'] = np.where(df_without_outliers['DR_QTY'] == df_without_outliers['AC_QTY'],
                                        df_without_outliers['UNIT_PRICE'] * df_without_outliers['DR_QTY'],
                                        df_without_outliers['UNIT_PRICE'] * df_without_outliers['AC_QTY'])
    df_without_outliers['TOTALAMOUNT'] = df_without_outliers['TOTALAMOUNT'].round(2)
    weightage_unit_price = 0.10  # Adjust the weightage as needed
    df_normalized_unit_price = reverse_normalize_by_group(df_without_outliers, 'ITEM_PROCESSED', 'UNIT_PRICE', weightage_unit_price)
    lead_time_weightage= .15
    df_normalized_lead_time = reverse_normalize_by_group(df_normalized_unit_price, 'ITEM_PROCESSED', 'LEAD_DAYS', lead_time_weightage)
    weightage_item_no_stock = 0.375
    df_normalized_no_item_stock = custom_normalize_neg_weight_single(df_normalized_lead_time, 'ITEM_NO_STOCK', weightage_item_no_stock)
    weightage_item_qc_not_ok = 0.375
    df_normalized = custom_normalize_neg_weight_single(df_normalized_no_item_stock, 'ITEM_QC_NOT_OK', weightage_item_qc_not_ok)
    df_normalized['OVERALL_SCORE'] = df_normalized[['NORM_UNIT_PRICE', 'NORM_LEAD_DAYS', 'NORM_ITEM_NO_STOCK', 'NORM_ITEM_QC_NOT_OK']].sum(axis=1).round(3)
    return df_normalized

def clean_item_name(item):
    import re
    # Remove special characters (excluding letters, digits, spaces, and hyphens)
    cleaned_item = re.sub(r'[^a-zA-Z0-9\s-]', ' ', item)
    # Replace multiple spaces with a single space
    cleaned_item = re.sub(r'\s+', ' ', cleaned_item)
    return cleaned_item.strip()

def reverse_normalize_by_group(df, group_column, column_name, weightage):
    
    if column_name not in df.columns or group_column not in df.columns:
        raise ValueError(f"Column '{column_name}' or '{group_column}' not found in the DataFrame.")

    reverse_normalize = lambda x: round(1 - ((x - x.min()) / max((x.max() - x.min()), 1)), 3)

    normalized_column_name = f'NORM_{column_name}'

    df[normalized_column_name] = df.groupby(group_column)[column_name].transform(
        lambda x: round(reverse_normalize(x) * weightage, 3)
    )
    return df

def get_vendor_rankings(df):
    # Assuming df is your DataFrame
    df['PO_APPROVED_MONTH'] = df['PO_APPROVED_MONTH'].astype(str)
    df['PO_APPROVED_YEAR'] = df['PO_APPROVED_YEAR'].astype(str)

    # Concatenate 'PO_APPROVED_YEAR' and 'PO_APPROVED_MONTH' to create a quarter column
    df['QUARTER'] = (df['PO_APPROVED_MONTH'].astype(int) - 1) // 3 + 1
    df['QUARTER'] = df['PO_APPROVED_YEAR'] + 'Q' + df['QUARTER'].astype(str)

    # Convert 'PO_APPROVED_YEAR' and 'PO_APPROVED_MONTH' to datetime for better sorting
    df['DATE'] = pd.to_datetime(df['PO_APPROVED_YEAR'] + '-' + df['PO_APPROVED_MONTH'])

    # Group by 'VENDOR', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH', 'DATE' and calculate the sum of the numeric representation
    # monthly_ranking = df.groupby(['VENDOR', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH', 'DATE'])['OVERALL_SCORE'].sum().reset_index()

    # # Calculate the maximum and minimum values for each time period
    # max_min_values = monthly_ranking.groupby(['PO_APPROVED_YEAR', 'PO_APPROVED_MONTH'])['OVERALL_SCORE'].agg(['max', 'min']).reset_index()

    # # Merge with monthly_ranking to have max and min values for each row
    # monthly_ranking = pd.merge(monthly_ranking, max_min_values, on=['PO_APPROVED_YEAR', 'PO_APPROVED_MONTH'], how='left', suffixes=('', '_MAX_MIN'))

    # # Create star levels based on max and min values
    # monthly_ranking['RATING'] = pd.qcut(
    #     monthly_ranking['OVERALL_SCORE'],
    #     q=[0, 0.2, 0.4, 0.6, 0.8, 1],
    #     labels=['★', '★★', '★★★', '★★★★', '★★★★★'],
    #     duplicates='drop'
    # )

    # Sort the results using the custom order
    # monthly_ranking = monthly_ranking.sort_values(by=['PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']).reset_index(drop=True)

    # Display the results for monthly ranking with stars
    # print("\nMonthly Ranking:")
    # print(tabulate(monthly_ranking[monthly_ranking['VENDOR'] == vendor_name][['VENDOR', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH', 'RATING']], headers='keys', tablefmt='pretty', showindex=False))

    # Group by 'VENDOR', 'QUARTER' and calculate the sum of the numeric representation
    # quarterly_ranking = df.groupby(['VENDOR', 'QUARTER'])['OVERALL_SCORE'].sum().reset_index()

    # # Calculate the maximum and minimum values for each time period
    # max_min_values = quarterly_ranking.groupby(['QUARTER'])['OVERALL_SCORE'].agg(['max', 'min']).reset_index()

    # # Merge with quarterly_ranking to have max and min values for each row
    # quarterly_ranking = pd.merge(quarterly_ranking, max_min_values, on=['QUARTER'], how='left', suffixes=('', '_MAX_MIN'))

    # # Create star levels based on max and min values
    # quarterly_ranking['RATING'] = pd.qcut(
    #     quarterly_ranking['OVERALL_SCORE'],
    #     q=[0, 0.2, 0.4, 0.6, 0.8, 1],
    #     labels=['★', '★★', '★★★', '★★★★', '★★★★★'],
    #     duplicates='drop'
    # )

    # Sort the results using the custom order
    # quarterly_ranking = quarterly_ranking.sort_values(by=['QUARTER']).reset_index(drop=True)

    # Display the results for quarterly ranking with stars
    # print("\nQuarterly Ranking:")
    # print(tabulate(quarterly_ranking[quarterly_ranking['VENDOR'] == vendor_name][['VENDOR', 'QUARTER', 'RATING']], headers='keys', tablefmt='pretty', showindex=False))

    # Group by 'VENDOR', 'PO_APPROVED_YEAR' and calculate the sum of the numeric representation
    yearly_ranking = df.groupby(['VENDOR','PORT_NAME'])['OVERALL_SCORE'].sum().reset_index()

    # Calculate the maximum and minimum values for each time period
    max_min_values = yearly_ranking.groupby(['PORT_NAME'])['OVERALL_SCORE'].agg(['max', 'min']).reset_index()

    # Merge with yearly_ranking to have max and min values for each row
    yearly_ranking = pd.merge(yearly_ranking, max_min_values, on=['PORT_NAME'], how='left', suffixes=('', '_MAX_MIN'))

    # Create star levels based on max and min values
    yearly_ranking['RATING'] = pd.qcut(
        yearly_ranking['OVERALL_SCORE'],
        q=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=['★', '★★', '★★★', '★★★★', '★★★★★'],
        duplicates='drop'
    )

    # Sort the results using the custom order
    yearly_ranking = yearly_ranking.sort_values(by=['PORT_NAME']).reset_index(drop=True)

    # Display the results for yearly ranking with stars
    print("\nYearly Ranking:")
    # print(tabulate(yearly_ranking[yearly_ranking['VENDOR'] == vendor_name][['VENDOR','RATING','PORT_NAME']], headers='keys', tablefmt='pretty', showindex=False))
    return yearly_ranking

def custom_normalize_neg_weight_single(df, column_name, weightage):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    normalized_column_name = f'NORM_{column_name}'

    # Assign values based on the specified conditions
    df[normalized_column_name] = df[column_name].apply(
        lambda x: round(-weightage if x != 0 else 0, 3)
    )
    return df

def supplier_evaluation(df,vendor):
    import re
    import numpy as np
    df_normalized =preprocessing_dataframe(df)
    df_normalized['item_price'] = df_normalized['PO_QTY'] * df_normalized['UNIT_PRICE']
    output_frame = pd.DataFrame() 
    df_normalized.dropna(subset=['ITEM_SECTION_1','ITEM_SECTION_2','UNIT_PRICE','PO_QTY','RECEIPT_DATE','PO_APPROVED_YEAR','PORT_NAME','UNIT_PRICE'],inplace=True)
    df_normalized.drop_duplicates(inplace=True)
    df_normalized['RECEIPT_DATE'] = pd.to_datetime(df_normalized['RECEIPT_DATE'],format='%d-%m-%Y %H:%M',errors='coerce')
    items_dict = {}
    for i in df_normalized['ITEM'].unique():
        k = re.sub(r'\W',' ',i).lower()
        lwr_k = re.sub(r'\s{2,}',' ',k).strip()
        items_dict[i] = lwr_k
    items_dict_rev = {v1:k1 for k1,v1 in items_dict.items()}
    df_normalized['ITEM_processed'] = df_normalized['ITEM'].replace(items_dict)
    df_normalized['ITEM_processed']
    # if item != None:
    #     item_proc = re.sub(r'\W',' ',item).lower()
    #     item_proc = re.sub(r'\s{2,}',' ',item_proc).strip()
    # df_normalized = df_normalized[(df_normalized['VENDOR'] == vendor)]
    df_normalized = df_normalized[df_normalized['VENDOR'] == vendor]

    for port in df_normalized['PORT_NAME'].value_counts().head(10).keys():
        df_subset = df_normalized[df_normalized['PORT_NAME'] == port]
        
        # po_qty_total = df_subset['PO_QTY'].sum()
        df_subset.sort_values(by='RECEIPT_DATE', ascending=False, inplace=True)
        # unit_price = list(df_subset['UNIT_PRICE'])[0]
        # unique_PO_CODE = df_subset['PO_CODE'].nunique()
        
        item = list(dict(df_subset['ITEM_processed'].value_counts().head(10)).keys())
        itm_unit_price = []
        itm_lead_time = []
        # itm_port = []
        itm_rec_date = []
        list_1 = []
        for itm1 in item:
            itm_unit_price.append(df_subset[df_subset['ITEM_processed']==itm1]['UNIT_PRICE'].apply(lambda x:np.max(x)).values[0])
            itm_lead_time.append(df_subset[df_subset['ITEM_processed']==itm1]['LEAD_DAYS'].apply(lambda x:np.mean(x)).values[0])
            list_1.append(df_subset[df_subset['ITEM_processed']==itm1]['item_price'].sum())
        itm_port = list(df_subset[df_subset['ITEM_processed']==itm1]['PORT_NAME'])[0]

            # itm_rec_date.append(list(df_subset[df_subset['ITEM_processed']==itm1]['RECEIPT_DATE'])[0])
            
        
        item = [items_dict_rev[ity] for ity in item]
        dff_sub = pd.DataFrame({'Vendor': [vendor], 'Port': ['p'], 'Mean_Lead_Time': ['c'], 'Item': [item], 'Total_Price per item': ['a']},index=[0])
        dff_sub.at[0,'Item'] = item
        # dff_sub.at[0,'max_unit_price'] = itm_unit_price

        dff_sub.at[0,'Mean_Lead_Time'] = itm_lead_time
        dff_sub.at[0,'Port'] = itm_port
        dff_sub.at[0,'Total_Price per item'] = list_1
        # dff_sub.at[0,'receipt date'] = itm_rec_date
        output_frame = pd.concat([output_frame,dff_sub]).head(20)
    output_frame['vendor_total_price_sum'] = [np.sum(xx) for xx in output_frame['Total_Price per item']]
    output_frame.sort_values(by='vendor_total_price_sum', inplace=True, ascending=False)
    output_frame.drop(columns=['vendor_total_price_sum'],inplace=True)
    output_frame.reset_index(drop=True, inplace=True)
    vendor_ratings_df = get_vendor_rankings(df_normalized)
    vendor_ratings_df = vendor_ratings_df[vendor_ratings_df['VENDOR']==vendor]
    vendor_ratings_df.rename(columns={'PORT_NAME':'Port','VENDOR':'Vendor'},inplace=True)
    # output_frame = pd.concat([output_frame,vendor_ratings_df[['PORT_NAME','RATING']]],join='left',axis=0)
    output_frame = pd.merge(output_frame,vendor_ratings_df,on = 'Port')
    output_frame = output_frame.rename(columns={'Vendor_x': 'Vendor'})
    output_frame.drop(columns=['OVERALL_SCORE','max','min','Vendor_y'],inplace=True)
    return output_frame
