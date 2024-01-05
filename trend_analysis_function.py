import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings("ignore")

def port_item_count_port(df,item_cat,itemsec1,itemsec2=None,portt=None):
    
    df = df.dropna(subset=['ITEM_CATEGORY','VENDOR_ID','VENDOR','PORT_ID','PORT_NAME','ITEM_SECTION_1','ITEM_SECTION_2','ITEM','UNIT_PRICE','PO_APPROVED_YEAR','RECEIPT_DATE'])
    df.drop_duplicates(inplace=True)
    items_dict = {}
    for i in df['ITEM'].unique():
        k = re.sub(r'\W',' ',i).lower()
        lwr_k = re.sub(r'\s{2,}',' ',k).strip()
        items_dict[i] = lwr_k
        
    df['ITEM_processed'] = df['ITEM'].replace(items_dict)
    
    outliers = df.groupby(['VENDOR', 'ITEM_processed','PO_APPROVED_YEAR'])['UNIT_PRICE'].apply(
    lambda x: x[(x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
                (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))])
    outliers = outliers.reset_index(level=[0, 1])
    outliers = pd.merge(outliers, df[['VENDOR', 'ITEM_processed', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']], how='left',
                        on=['VENDOR', 'ITEM_processed'])
    outliers_df=outliers.drop_duplicates()
    merged_df = pd.merge(df, outliers_df, how='left', indicator=True)
    df_without_outliers = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_without_outliers.drop_duplicates(inplace=True)
 
    df=df_without_outliers   
    df['RECEIPT_DATE']=pd.to_datetime(df['RECEIPT_DATE'],format='%d-%m-%Y %H:%M',errors='coerce')
    # item_dict={}
    df=df[df['ITEM_CATEGORY']==item_cat]
    if len(df)>0:
        
        
        if portt!=None:
           
            df=df[(df['ITEM_SECTION_1']==itemsec1) ] 
            if len(df)>0:
                df=df[df['PORT_NAME']==portt] 
                if len(df)>0:
                
                    if itemsec2!=None:
                        # df=df[df['PO_APPROVED_YEAR']==year]
                        # df.sort_values('RECEIPT_DATE',ascending=False,inplace=True)
                        df=df[df['ITEM_SECTION_2']==itemsec2]
                        
                        
                        if len(df)>0:                           
                            item_counts = df['ITEM_processed'].value_counts().head(20)
                            top_items=pd.DataFrame()
                            
                            
                            
                            
                            for i in item_counts.keys():
                                item_df=df[df['ITEM_processed']==i].head(1)
                                
                                min_unit_price = df[df['ITEM_processed']==i]['UNIT_PRICE'].min()
                                
                                item_df['MIN_UNIT_PRICE'] = min_unit_price
                                
                                
                                top_items=pd.concat([top_items,item_df[['ITEM_CATEGORY','VENDOR_ID','VENDOR','PORT_ID','PORT_NAME','ITEM_SECTION_1','ITEM_SECTION_2','ITEM','MIN_UNIT_PRICE','RECEIPT_DATE']]])
                                # item_dict[portt]=top_items
                            return top_items                             
                    else:                           
                        item_counts = df['ITEM_processed'].value_counts().head(20)
                        top_items=pd.DataFrame()
                        
                        
                        
                        for i in item_counts.keys():
                            
                            item_df=df[df['ITEM_processed']==i].head(1)
                            min_unit_price = df[df['ITEM_processed']==i]['UNIT_PRICE'].min()
                            item_df['MIN_UNIT_PRICE'] = min_unit_price
                            
                            top_items=pd.concat([top_items,item_df[['ITEM_CATEGORY','VENDOR_ID','VENDOR','PORT_ID','PORT_NAME','ITEM_SECTION_1','ITEM_SECTION_2','ITEM','MIN_UNIT_PRICE','RECEIPT_DATE']]])
                            # item_dict[portt]=top_items

                        return top_items    
        else:
            # item_dict={}
            item_df = pd.DataFrame()
        
            df=df[(df['ITEM_SECTION_1']==itemsec1)] 
            if len(df)>0:
                if itemsec2!=None:
                    df=df[df['ITEM_SECTION_2']==itemsec2]
                     # df=df[df['PO_APPROVED_YEAR']==year]
                    # df.sort_values('RECEIPT_DATE',ascending=False,inplace=True)
                    
                    if len(df)>0:
                        for port in df['PORT_NAME'].unique():
                            port_subset = df[df['PORT_NAME'] == port]
                            
                            item_counts = port_subset['ITEM_processed'].value_counts().head(20)

                            top_items=pd.DataFrame()
                            
                            
                            
                            
                            for i in item_counts.keys():
                                port_subset_item=port_subset[port_subset['ITEM_processed']==i].head(1)
                                min_unit_price = port_subset[port_subset['ITEM_processed']==i]['UNIT_PRICE'].min()
                                port_subset_item['MIN_UNIT_PRICE'] = min_unit_price
                                top_items=pd.concat([top_items,port_subset_item[['ITEM_CATEGORY','VENDOR_ID','VENDOR','PORT_ID','PORT_NAME','ITEM_SECTION_1','ITEM_SECTION_2','ITEM','MIN_UNIT_PRICE','RECEIPT_DATE']]])
                            # item_dict[port]=top_items
                            item_df = pd.concat([item_df,top_items])
                        return item_df 
                else:
                    for port in df['PORT_NAME'].unique():
                        port_subset = df[df['PORT_NAME'] == port]
                        item_counts = port_subset['ITEM_processed'].value_counts().head(20)
                        top_items=pd.DataFrame()
                        
                        
                        
                        
                        
                        for i in item_counts.keys():
                            port_subset_item=port_subset[port_subset['ITEM_processed']==i].head(1)
                            min_unit_price = port_subset[port_subset['ITEM_processed']==i]['UNIT_PRICE'].min()
                            port_subset_item['MIN_UNIT_PRICE'] = min_unit_price
                            top_items=pd.concat([top_items,port_subset_item[['ITEM_CATEGORY','VENDOR_ID','VENDOR','PORT_ID','PORT_NAME','ITEM_SECTION_1','ITEM_SECTION_2','ITEM','MIN_UNIT_PRICE','RECEIPT_DATE']]])
                    # item_dict[port]=top_items
                        item_df = pd.concat([item_df,top_items])
                    
                
                    return item_df 
                
