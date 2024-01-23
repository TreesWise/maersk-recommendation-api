import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Reverse Normalization Function for giving more weightage to items with low unit price and lead days
def reverse_normalize_by_group(df, group_columns, column_name, weightage):
    if column_name not in df.columns or any(col not in df.columns for col in group_columns):
        raise ValueError(f"Column '{column_name}' or one of the group columns not found in the DataFrame.")
    
    # Create a unique group identifier based on the specified group columns
    df['TEMP_GROUP'] = df[group_columns].astype(str).agg('-'.join, axis=1)
    
    # Determine the max and min for each group
    group_max = df.groupby('TEMP_GROUP')[column_name].transform('max').rename(f'MAX_{column_name}')
    group_min = df.groupby('TEMP_GROUP')[column_name].transform('min').rename(f'MIN_{column_name}')
    
    # Calculate the reverse normalized values
    reverse_normalize = lambda x: round(1 - ((x - x.min()) / max((x.max() - x.min()), 1)), 3)
    normalized_column_name = f'NORM_{column_name}'
    df[normalized_column_name] = df.groupby('TEMP_GROUP')[column_name].transform(
        lambda x: reverse_normalize(x) * weightage if x.max() != x.min() else weightage
    )
    
    # Add the max and min columns to the DataFrame
    df[f'MAX_{column_name}'] = group_max
    df[f'MIN_{column_name}'] = group_min
    
    # Drop the temporary group identifier column
    df.drop('TEMP_GROUP', axis=1, inplace=True)
    
    return df

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Normal Normalization function to assign scores ranging from 0-1 depending on the min and max 
def normalize_by_group(df, group_columns, column_name, weightage):
    if column_name not in df.columns or any(col not in df.columns for col in group_columns):
        raise ValueError(f"Column '{column_name}' or one of the group columns not found in the DataFrame.")
    
    # Create a unique group identifier based on the specified group columns
    df['TEMP_GROUP'] = df[group_columns].astype(str).agg('-'.join, axis=1)
    
    # Determine the max and min for each group
    group_max = df.groupby('TEMP_GROUP')[column_name].transform('max').rename(f'MAX_{column_name}')
    group_min = df.groupby('TEMP_GROUP')[column_name].transform('min').rename(f'MIN_{column_name}')
    
    # Calculate the normalized values
    normalize = lambda x: round(((x - x.min()) / max((x.max() - x.min()), 1)), 3)
    normalized_column_name = f'NORM_{column_name}'
    df[normalized_column_name] = df.groupby('TEMP_GROUP')[column_name].transform(
        lambda x: normalize(x) * weightage if x.max() != x.min() else weightage
    )
    
    # Add the max and min columns to the DataFrame
    df[f'MAX_{column_name}'] = group_max
    df[f'MIN_{column_name}'] = group_min
    
    # Drop the temporary group identifier column
    df.drop('TEMP_GROUP', axis=1, inplace=True)

    return df

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Custom normalization function to assign scores for suppliers without stock/ low quality
def custom_normalize_neg_weight_single(df, column_name, weightage):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    normalized_column_name = f'NORM_{column_name}'
    # Assign values based on the specified conditions
    df[normalized_column_name] = df[column_name].apply(
        lambda x: round(weightage if x == 0 else 0, 3)
    )
    return df

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def predict_overall_score(df_input,item,port,model,preprocessor,po_qty):
    filtered_df = df_input[['PORT', 'VENDOR', 'ITEM','MEAN_LEAD_DAYS', 'COUNT_PER_VENDOR_PORT', 'DIVERSITY_OF_VENDOR']]
    # Load the pre-trained model
    filtered_df = filtered_df[(filtered_df['ITEM'] == item) & 
                (filtered_df['PORT'] == port)]
    filtered_df.drop_duplicates(inplace=True)
    print("filtered_df_length",len(filtered_df))
    if len(filtered_df)!=0:
        filtered_df['PO_QTY']= po_qty
        # print(filtered_df.dtypes)
        model = model
        preprocessor = preprocessor
        # Apply preprocessing to the input DataFrame
        X = filtered_df[['PORT', 'VENDOR', 'ITEM', 'PO_QTY', 'MEAN_LEAD_DAYS', 'COUNT_PER_VENDOR_PORT', 'DIVERSITY_OF_VENDOR']]
        X_processed = preprocessor.transform(X)
        X_processed = X_processed.astype('float32')
        # X_processed.shape
        # Make predictions using the trained model
        y_pred = model.predict(X_processed)
        # Store the predictions in a new column in the input DataFrame
        filtered_df['PREDICTED_OVERALL_SCORE'] = y_pred.flatten()

        # Categorize the predictions
        # Define your thresholds here. For example:
        low_threshold = 0.45
        high_threshold = 0.85

        # Handling the edge cases
        num_vendors = len(filtered_df['VENDOR'].unique())
        if num_vendors > 2:
            filtered_df['EVALUATION SCORE'] = pd.cut(filtered_df['PREDICTED_OVERALL_SCORE'], 
                                                    bins=[0, low_threshold, high_threshold, 1], 
                                                    labels=['Low', 'Medium', 'High'], 
                                                    include_lowest=True)
        elif num_vendors == 2:
            # For two vendors, categorize directly as Low or High
            filtered_df['EVALUATION SCORE'] = pd.cut(filtered_df['PREDICTED_OVERALL_SCORE'], 
                                                    bins=[0, 0.5, 1], 
                                                    labels=['Low', 'High'], 
                                                    include_lowest=True)
        else:
            # For a single vendor, assign Medium or another appropriate category
            filtered_df['EVALUATION SCORE'] = 'Medium'
        df_copy=filtered_df[['VENDOR','ITEM','EVALUATION SCORE']]
        return df_copy
    else:
        print("error")
        dicterror= {"error":"norecord found for the given input"}
        errordf= pd.DataFrame(dicterror,index=[0])
        return errordf

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def preprocessing(df):
    section_from_item = ['PROVISION TOTAL', 'PROVISION', 'BONDED STORES']
    item_section_from_item = ['TYPE-IN']
    df = df[~df['ITEM'].isin(section_from_item)]
    df = df[~df['ITEM_SECTION_1'].isin(item_section_from_item)]
    df = df[['PORT', 'VENDOR', 'ITEM', 'LEAD_DAYS','PO_ID','PO_QTY', 'DR_QTY', 'AC_QTY', 'UNIT_PRICE', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']]
    df.dropna(subset=['UNIT_PRICE'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    df['LEAD_DAYS'] = df['LEAD_DAYS'].abs()
    df.dropna(subset=['DR_QTY'], inplace=True)
    df.dropna(subset=['AC_QTY'], inplace=True)
    df['ITEM_NO_STOCK']=df['PO_QTY']-df['DR_QTY']
    df['ITEM_QC_NOT_OK']= df['DR_QTY']-df['AC_QTY']
    filtered_df = df.dropna(subset=['ITEM'])
    filtered_df.reset_index(drop=True,inplace=True)
    outliers = filtered_df.groupby(['VENDOR', 'ITEM','PO_APPROVED_YEAR','PORT'])['UNIT_PRICE'].apply(
    lambda x: x[(x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
                (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))])
    outliers = outliers.reset_index(level=[0, 1])
    outliers = pd.merge(outliers, filtered_df[['VENDOR', 'ITEM', 'PO_APPROVED_YEAR', 'PO_APPROVED_MONTH']], how='left',
                        on=['VENDOR', 'ITEM'])
    outliers_df=outliers.drop_duplicates()
    merged_df = pd.merge(filtered_df, outliers_df, how='left', indicator=True)
    df_without_outliers = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_without_outliers.drop_duplicates(inplace=True)
    # print('columns',df_without_outliers.columns)
    weightage_unit_price = 0.20  # Adjust the weightage as needed
    df_without_outliers['COUNT_PER_VENDOR_PORT'] = df_without_outliers.groupby(['VENDOR', 'PORT'])['PO_ID'].transform('count')
    df_without_outliers['DIVERSITY_OF_VENDOR'] = df_without_outliers.groupby('VENDOR')['PORT'].transform('nunique')
    min_value = df_without_outliers['DIVERSITY_OF_VENDOR'].min()
    range_value = df_without_outliers['DIVERSITY_OF_VENDOR'].max() - min_value
    df_without_outliers['NORM_DIVERSITY_OF_VENDOR'] = ((df_without_outliers['DIVERSITY_OF_VENDOR'] - min_value) / range_value) * 0.10
    # df_without_outliers.to_csv("No_Outliers.csv",index=False)
    df_normalized_unit_price = reverse_normalize_by_group(df_without_outliers, ['PORT', 'ITEM'], 'UNIT_PRICE', weightage_unit_price)
    lead_time_weightage= 0.20
    df_normalized_lead_time = reverse_normalize_by_group(df_normalized_unit_price, ['PORT', 'ITEM'], 'LEAD_DAYS', lead_time_weightage)
    weightage_item_no_stock = 0.20
    df_normalized_no_item_stock = custom_normalize_neg_weight_single(df_normalized_lead_time, 'ITEM_NO_STOCK', weightage_item_no_stock)
    weightage_item_qc_not_ok = 0.20
    df_normalized_qc_not_ok = custom_normalize_neg_weight_single(df_normalized_no_item_stock, 'ITEM_QC_NOT_OK', weightage_item_qc_not_ok)
    weightage_vendor_purchase_count=.10
    df_normalized=normalize_by_group(df_normalized_qc_not_ok,['PORT','VENDOR'],'COUNT_PER_VENDOR_PORT',weightage_vendor_purchase_count)
    df_normalized['OVERALL_SCORE'] = df_normalized[['NORM_UNIT_PRICE', 'NORM_LEAD_DAYS', 'NORM_ITEM_NO_STOCK', 'NORM_ITEM_QC_NOT_OK','NORM_COUNT_PER_VENDOR_PORT','NORM_DIVERSITY_OF_VENDOR']].sum(axis=1).round(3)
    df_normalized['MEAN_LEAD_DAYS'] = df_normalized.groupby(['VENDOR', 'ITEM', 'PORT'])['LEAD_DAYS'].transform('mean')
    return filtered_df
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------