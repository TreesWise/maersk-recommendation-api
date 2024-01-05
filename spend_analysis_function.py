# 1.Finding the top 20 Vendor's based on total spend amount on purchase order's after grouping by (item section1 and item section 2,item name(optional)) .
import re
import pandas as pd
import numpy as np
def spend_analysis(df, item_cat, item_sec1, item_sec2, item=None):
    df.dropna(subset=['ITEM_SECTION_1','ITEM_SECTION_2','UNIT_PRICE','PO_QTY','RECEIPT_DATE','PO_APPROVED_YEAR','PORT_NAME'],inplace=True)
    df.drop_duplicates(inplace=True)
    df['RECEIPT_DATE'] = pd.to_datetime(df['RECEIPT_DATE'],format='%d-%m-%Y %H:%M',errors='coerce')
    items_dict = {}
    for i in df['ITEM'].unique():
        k = re.sub(r'\W',' ',i).lower()
        lwr_k = re.sub(r'\s{2,}',' ',k).strip()
        items_dict[i] = lwr_k
    items_dict_rev = {v1:k1 for k1,v1 in items_dict.items()}
    df['ITEM_processed'] = df['ITEM'].replace(items_dict)
    df['ITEM_processed']
 
    outliers = df.groupby(['VENDOR', 'ITEM_processed','PO_APPROVED_YEAR'])['UNIT_PRICE'].apply(
    lambda x: x[(x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
                (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))])
    outliers = outliers.reset_index(level=[0, 1])
    outliers = pd.merge(outliers, df[['VENDOR', 'ITEM_processed', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']], how='left',
                        on=['VENDOR', 'ITEM_processed'])
    outliers_df=outliers#.drop_duplicates()
    merged_df = pd.merge(df, outliers_df, how='left', indicator=True)
    df_without_outliers = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_without_outliers.drop_duplicates(inplace=True)
    df = df_without_outliers
 
   
    if item != None:
        item_proc = re.sub(r'\W',' ',item).lower()
        item_proc = re.sub(r'\s{2,}',' ',item_proc).strip()
 
    df['item_price'] = df['PO_QTY'] * df['UNIT_PRICE']
 
    df = df[(df['ITEM_CATEGORY'] == item_cat) & (df['ITEM_SECTION_1'] == item_sec1) & (df['ITEM_SECTION_2'] == item_sec2) & (df['ITEM_processed'] == item_proc if item is not None else True)]
 
    if len(df) > 0:
        output_frame = pd.DataFrame()
        if item is not None:
            for vendor in df['VENDOR'].unique():
                df_subset = df[df['VENDOR'] == vendor]
                vendor_item_total = df_subset['item_price'].sum()
                po_qty_total = df_subset['PO_QTY'].sum()
                df_subset.sort_values(by='RECEIPT_DATE', ascending=False, inplace=True)
                unique_PO_CODE = df_subset[df_subset['ITEM_processed'] == item_proc]['PO_CODE'].nunique()
                # unit_price = df_subset['UNIT_PRICE'].max()
                port = list(df_subset[df_subset['ITEM_processed'] == item_proc]['PORT_NAME'])[0]
                max_unit_price = df_subset[df_subset['ITEM_processed'] == item_proc]['UNIT_PRICE'].max()
                avg_lead_time = df_subset[df_subset['ITEM_processed'] == item_proc]['LEAD_DAYS'].median()
                receipt_date = list(df_subset[df_subset['ITEM_processed'] == item_proc]['RECEIPT_DATE'])[0]
                dff_sub = pd.DataFrame({'vendor': vendor, 'item section 1': item_sec1, 'item section 2': item_sec2, 'item': [item],'Transactional_volume':unique_PO_CODE, 'max_unit_price': [max_unit_price],  'port': [port], 'avg lead time': [avg_lead_time],
                                                                        'receipt date': [receipt_date], 'vendor_total_price': vendor_item_total, 'vendor_po_qty_total': po_qty_total}, index=[1])
                output_frame = pd.concat([output_frame, dff_sub]).head(20)
            output_frame.sort_values(by='vendor_total_price', inplace=True, ascending=False)
            output_frame.reset_index(drop=True, inplace=True)
            return output_frame
        else:
            for vendor in df['VENDOR'].unique():
                df_subset = df[df['VENDOR'] == vendor]
                vendor_item_total = df_subset['item_price'].sum()
                po_qty_total = df_subset['PO_QTY'].sum()
                df_subset.sort_values(by='RECEIPT_DATE', ascending=False, inplace=True)
                # unit_price = list(df_subset['UNIT_PRICE'])[0]
                unique_PO_CODE = df_subset['PO_CODE'].nunique()
                # print(df_subset['PO_CODE'].unique())
                # port = list(df_subset['PORT_NAME'])[0]
                # avg_unit_price = df_subset['UNIT_PRICE'].median()
                # avg_lead_time = df_subset['LEAD_DAYS'].median()
                # receipt_date = list(df_subset['RECEIPT_DATE'])[0]
                # item =items_dict_rev[list(df_subset['ITEM_processed'])[0]]
                item = list(dict(df_subset['ITEM_processed'].value_counts().head(10)).keys())
                itm_unit_price = []
                itm_lead_time = []
                itm_port = []
                itm_rec_date = []
                for itm1 in item:
                    itm_unit_price.append(df_subset[df_subset['ITEM_processed']==itm1]['UNIT_PRICE'].apply(lambda x:np.max(x)).values[0])
                    itm_lead_time.append(df_subset[df_subset['ITEM_processed']==itm1]['LEAD_DAYS'].apply(lambda x:np.median(x)).values[0])
                    itm_port.append(list(df_subset[df_subset['ITEM_processed']==itm1]['PORT_NAME'])[0])
                    itm_rec_date.append(list(df_subset[df_subset['ITEM_processed']==itm1]['RECEIPT_DATE'])[0])
                item = [items_dict_rev[ity] for ity in item]
                dff_sub = pd.DataFrame({'vendor': vendor, 'item section 1': item_sec1, 'item section 2': item_sec2, 'item':'a','Transactional_volume':unique_PO_CODE,
                                                                      'max_unit_price': 'b', 'port': 'p', 'avg_lead_time': 'c', 'receipt date': 'd',
                                                                      'vendor_total_price': vendor_item_total, 'vendor_po_qty_total': po_qty_total},index=[0])
                dff_sub.at[0,'item'] = item
                dff_sub.at[0,'max_unit_price'] = itm_unit_price
                dff_sub.at[0,'avg_lead_time'] = itm_lead_time
                dff_sub.at[0,'port'] = itm_port
                dff_sub.at[0,'receipt date'] = itm_rec_date
                output_frame = pd.concat([output_frame,dff_sub]).head(20)
            output_frame.sort_values(by='vendor_total_price', inplace=True, ascending=False)
            output_frame.reset_index(drop=True, inplace=True)
            return output_frame