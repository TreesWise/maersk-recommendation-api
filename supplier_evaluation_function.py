import pandas as pd
import numpy as np
import re

def custom_normalize_neg_weight_single(df, column_name, weightage):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    normalized_column_name = f'NORM_{column_name}'

    # Assign values based on the specified conditions
    df[normalized_column_name] = df[column_name].apply(
        lambda x: round(-weightage if x != 0 else 0, 3)
    )
    return df


def reverse_normalize_by_group(df, group_columns, column_name, weightage):
    if column_name not in df.columns or any(col not in df.columns for col in group_columns):
        raise ValueError(f"Column '{column_name}' or one of the group columns not found in the DataFrame.")

    reverse_normalize = lambda x: round(1 - ((x - x.min()) / max((x.max() - x.min()), 1)), 3)

    normalized_column_name = f'NORM_{column_name}'

    # Concatenate values of group columns to create a new temporary column for grouping
    df['TEMP_GROUP'] = df[group_columns].astype(str).agg('-'.join, axis=1)

    # Apply reverse normalization with weightage to all groups
    df[normalized_column_name] = df.groupby('TEMP_GROUP')[column_name].transform(
        lambda x: reverse_normalize(x) * weightage if len(x) > 1 else weightage
    )

    # Drop the temporary column after normalization
    df.drop('TEMP_GROUP', axis=1, inplace=True)

    return df



def clean_item_name(item):

    # Remove special characters (excluding letters, digits, spaces, and hyphens)
    cleaned_item = re.sub(r'[^a-zA-Z0-9\s-]', ' ', item)
    # Replace multiple spaces with a single space
    cleaned_item = re.sub(r'\s+', ' ', cleaned_item)
    return cleaned_item.strip()


def preprocessing_dataframe(df):
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
    df_normalized_unit_price = reverse_normalize_by_group(df_without_outliers, ['PORT_NAME', 'ITEM_PROCESSED'], 'UNIT_PRICE', weightage_unit_price)
    lead_time_weightage= .15
    df_normalized_lead_time = reverse_normalize_by_group(df_normalized_unit_price, ['PORT_NAME', 'ITEM_PROCESSED'], 'LEAD_DAYS', lead_time_weightage)
    weightage_item_no_stock = 0.375
    df_normalized_no_item_stock = custom_normalize_neg_weight_single(df_normalized_lead_time, 'ITEM_NO_STOCK', weightage_item_no_stock)
    weightage_item_qc_not_ok = 0.375
    df_normalized = custom_normalize_neg_weight_single(df_normalized_no_item_stock, 'ITEM_QC_NOT_OK', weightage_item_qc_not_ok)
    df_normalized['OVERALL_SCORE'] = df_normalized[['NORM_UNIT_PRICE', 'NORM_LEAD_DAYS', 'NORM_ITEM_NO_STOCK', 'NORM_ITEM_QC_NOT_OK']].sum(axis=1).round(3)
    return df_normalized

import pandas as pd

def get_vendor_ranking(df):
    # Assuming df is your DataFrame

    # Convert 'OVERALL_SCORE' to numeric (in case it's not already)
    df['OVERALL_SCORE'] = pd.to_numeric(df['OVERALL_SCORE'], errors='coerce')
    df['RATING'] = ''
    # Group by 'PORT_NAME' and 'VENDOR' and calculate the sum of OVERALL_SCORE and the total number of orders for each vendor within a given port
    vendor_sum = df.groupby(['PORT_NAME', 'VENDOR']).agg(
        OVERALL_SCORE_SUM=('OVERALL_SCORE', 'sum'),
        VENDOR_COUNT=('VENDOR', 'count')
    ).reset_index()    

    # Sort DataFrame by 'PORT_NAME' and 'OVERALL_SCORE' in descending order
    vendor_overall_score_sum = vendor_sum.sort_values(by=['PORT_NAME', 'OVERALL_SCORE_SUM'], ascending=[True, False])
    
    # Create a 'RANK' column within each port group
    vendor_overall_score_sum['ORDER_RANK'] = vendor_overall_score_sum.groupby(['PORT_NAME'])['OVERALL_SCORE_SUM'].rank(ascending=False, method='dense')
    vendor_overall_score_sum['VENDOR_COUNT_RANK'] = vendor_overall_score_sum.groupby(['PORT_NAME'])['VENDOR_COUNT'].rank(ascending=False, method='dense')
    vendor_overall_score_sum = vendor_overall_score_sum.drop(columns=['OVERALL_SCORE_SUM', 'VENDOR_COUNT'])

    # Calculate combined percentile
    vendor_overall_score_sum['COMBINED_RANK'] = (vendor_overall_score_sum['ORDER_RANK'] + vendor_overall_score_sum['VENDOR_COUNT_RANK']) / 2
    vendor_overall_score_sum['COMBINED_PERCENTILE'] = vendor_overall_score_sum.groupby(['PORT_NAME'])['COMBINED_RANK'].transform(
        lambda x: x.rank(ascending=False, method='dense') / x.nunique()
    )

    # Map the percentile to star rating
    def get_star_rating(percentile):
        stars = min(int(percentile * 5) + 1, 5)  # Map percentile to stars, cap at 5 stars
        return '★' * stars

    # Apply the star rating logic to the main DataFrame
    df = pd.merge(df, vendor_overall_score_sum[['PORT_NAME', 'VENDOR', 'COMBINED_PERCENTILE','ORDER_RANK','VENDOR_COUNT_RANK']], on=['PORT_NAME', 'VENDOR'], how='left')
    df['RATING'] = df['COMBINED_PERCENTILE'].apply(get_star_rating)

    # Drop temporary columns if needed
    # df = df.drop(columns=['COMBINED_PERCENTILE', 'COMBINED_RANK', 'ORDER_RANK', 'VENDOR_COUNT_RANK'])
    df=df[['PORT_NAME','VENDOR','RATING']]
    return df


#  5 seconds slower in execution
def supplier_evaluation(df,vendor):
    import re
    import numpy as np
    df_normalized =preprocessing_dataframe(df)
    vendor_ratings_df = get_vendor_ranking(df_normalized)
    df_normalized['item_price'] = df_normalized['PO_QTY'] * df_normalized['UNIT_PRICE']
    output_frame = pd.DataFrame() 
    df_normalized.dropna(subset=['ITEM_SECTION_1','ITEM_SECTION_2','UNIT_PRICE','PO_QTY','RECEIPT_DATE','PO_APPROVED_YEAR','PORT_NAME','UNIT_PRICE'],inplace=True)
    df_normalized.drop_duplicates(inplace=True)
    df_normalized['RECEIPT_DATE'] = pd.to_datetime(df_normalized['RECEIPT_DATE'],format='%d-%m-%Y %H:%M',errors='coerce')
    # for i in df_normalized['ITEM'].unique():
    #     k = re.sub(r'\W',' ',i).lower()
    #     lwr_k = re.sub(r'\s{2,}',' ',k).strip()
    #     items_dict[i] = lwr_k
    items_dict_rev = {df_normalized.loc[row[0],'ITEM_PROCESSED']:df_normalized.loc[row[0],'ITEM'] for row in df_normalized.iterrows()}
    # df_normalized['ITEM_processed'] = df_normalized['ITEM'].replace(items_dict)
    # df_normalized['ITEM_processed']
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
        
        item = list(dict(df_subset['ITEM_PROCESSED'].value_counts().head(10)).keys())
        itm_unit_price = []
        itm_lead_time = []
        # itm_port = []
        itm_rec_date = []
        list_1 = []
        for itm1 in item:
            itm_unit_price.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['UNIT_PRICE'].apply(lambda x:np.max(x)).values[0])
            itm_lead_time.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['LEAD_DAYS'].apply(lambda x:np.mean(x)).values[0])
            list_1.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['item_price'].sum())
        itm_port = list(df_subset[df_subset['ITEM_PROCESSED']==itm1]['PORT_NAME'])[0]

            # itm_rec_date.append(list(df_subset[df_subset['ITEM_processed']==itm1]['RECEIPT_DATE'])[0])        
        item = [items_dict_rev[ity] for ity in item]
        dff_sub = pd.DataFrame({'Vendor': [vendor], 'Port': ['p'], 'Mean_Lead_Time': ['c'], 'Item': [item], 'Total_Price per item': ['a']},index=[0])
        dff_sub.at[0,'Item'] = item
        # dff_sub.at[0,'max_unit_price'] = itm_unit_price
        dff_sub.at[0,'Mean_Lead_Time'] = itm_lead_time
        dff_sub.at[0,'Port'] = itm_port
        dff_sub.at[0,'Total_Price per item'] = list_1
        # dff_sub.at[0,'receipt date'] = itm_rec_date
        output_frame = pd.concat([output_frame,dff_sub])#.head(20)
    output_frame['vendor_total_price_sum'] = [np.sum(xx) for xx in output_frame['Total_Price per item']]
    output_frame.sort_values(by='vendor_total_price_sum', inplace=True, ascending=False)
    output_frame.drop(columns=['vendor_total_price_sum'],inplace=True)
    output_frame.reset_index(drop=True, inplace=True)
    vendor_ratings_df.rename(columns={'PORT_NAME':'Port','VENDOR':'Vendor'},inplace=True)
    for irow in range(len(output_frame)):
        output_frame.loc[irow,'Rating'] = list(vendor_ratings_df[(vendor_ratings_df['Vendor']==output_frame.loc[irow,'Vendor'])&(vendor_ratings_df['Port']==output_frame.loc[irow,'Port'])]['RATING'])[0]
    return output_frame.to_dict(orient='records')


#  5 seconds faster in execution
# def supplier_evaluation(df,vendor):
#     import re
#     import numpy as np
#     df_normalized =preprocessing_dataframe(df)
#     vendor_ratings_df = get_vendor_ranking(df_normalized)
#     df_normalized['item_price'] = df_normalized['PO_QTY'] * df_normalized['UNIT_PRICE']
#     output_frame = pd.DataFrame() 
#     df_normalized.dropna(subset=['ITEM_SECTION_1','ITEM_SECTION_2','UNIT_PRICE','PO_QTY','RECEIPT_DATE','PO_APPROVED_YEAR','PORT_NAME','UNIT_PRICE'],inplace=True)
#     df_normalized.drop_duplicates(inplace=True)
#     df_normalized['RECEIPT_DATE'] = pd.to_datetime(df_normalized['RECEIPT_DATE'],format='%d-%m-%Y %H:%M',errors='coerce')
#     items_dict = {}
#     for i in df_normalized['ITEM'].unique():
#         k = re.sub(r'\W',' ',i).lower()
#         lwr_k = re.sub(r'\s{2,}',' ',k).strip()
#         items_dict[i] = lwr_k
#     items_dict_rev = {v1:k1 for k1,v1 in items_dict.items()}
#     df_normalized['ITEM_PROCESSED'] = df_normalized['ITEM'].replace(items_dict)
#     # df_normalized['ITEM_processed']
#     # if item != None:
#     #     item_proc = re.sub(r'\W',' ',item).lower()
#     #     item_proc = re.sub(r'\s{2,}',' ',item_proc).strip()
#     # df_normalized = df_normalized[(df_normalized['VENDOR'] == vendor)]
#     df_normalized = df_normalized[df_normalized['VENDOR'] == vendor]
#     for port in df_normalized['PORT_NAME'].value_counts().head(10).keys():
#         df_subset = df_normalized[df_normalized['PORT_NAME'] == port]
        
#         # po_qty_total = df_subset['PO_QTY'].sum()
#         df_subset.sort_values(by='RECEIPT_DATE', ascending=False, inplace=True)
#         # unit_price = list(df_subset['UNIT_PRICE'])[0]
#         # unique_PO_CODE = df_subset['PO_CODE'].nunique()
        
#         item = list(dict(df_subset['ITEM_PROCESSED'].value_counts().head(10)).keys())
#         itm_unit_price = []
#         itm_lead_time = []
#         # itm_port = []
#         itm_rec_date = []
#         list_1 = []
#         for itm1 in item:
#             itm_unit_price.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['UNIT_PRICE'].apply(lambda x:np.max(x)).values[0])
#             itm_lead_time.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['LEAD_DAYS'].apply(lambda x:np.mean(x)).values[0])
#             list_1.append(df_subset[df_subset['ITEM_PROCESSED']==itm1]['item_price'].sum())
#         itm_port = list(df_subset[df_subset['ITEM_PROCESSED']==itm1]['PORT_NAME'])[0]

#             # itm_rec_date.append(list(df_subset[df_subset['ITEM_processed']==itm1]['RECEIPT_DATE'])[0])        
#         item = [items_dict_rev[ity] for ity in item]
#         dff_sub = pd.DataFrame({'Vendor': [vendor], 'Port': ['p'], 'Mean_Lead_Time': ['c'], 'Item': [item], 'Total_Price per item': ['a']},index=[0])
#         dff_sub.at[0,'Item'] = item
#         # dff_sub.at[0,'max_unit_price'] = itm_unit_price
#         dff_sub.at[0,'Mean_Lead_Time'] = itm_lead_time
#         dff_sub.at[0,'Port'] = itm_port
#         dff_sub.at[0,'Total_Price per item'] = list_1
#         # dff_sub.at[0,'receipt date'] = itm_rec_date
#         output_frame = pd.concat([output_frame,dff_sub])#.head(20)
#     output_frame['vendor_total_price_sum'] = [np.sum(xx) for xx in output_frame['Total_Price per item']]
#     output_frame.sort_values(by='vendor_total_price_sum', inplace=True, ascending=False)
#     output_frame.drop(columns=['vendor_total_price_sum'],inplace=True)
#     output_frame.reset_index(drop=True, inplace=True)
#     vendor_ratings_df.rename(columns={'PORT_NAME':'Port','VENDOR':'Vendor'},inplace=True)
#     for irow in range(len(output_frame)):
#         output_frame.loc[irow,'Rating'] = list(vendor_ratings_df[(vendor_ratings_df['Vendor']==output_frame.loc[irow,'Vendor'])&(vendor_ratings_df['Port']==output_frame.loc[irow,'Port'])]['RATING'])[0]
#     return output_frame.to_dict(orient='records')