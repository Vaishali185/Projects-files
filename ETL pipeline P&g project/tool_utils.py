# Rejects Tool specific functions

import pandas as pd

from sud_tools import *
from sud_utils import *

from typing import *


def fix_historian_counters(extract_df: pd.DataFrame, qty_column_name: str = 'rejects_qty') -> pd.DataFrame:
    """Fixing the counters tags extracted from historian by 
    1. Removing cases where the counter values were not logged (NaNs) or negative
    2. Removing cases, where counter had dropped down in between two resets (historian outage issue)
    3. Removing cases where the counter was not changing
    4. Removing cases where the counter reports zero
    
    
    """
    
    print(f'TRANSFORM: Shape of original paking rejects data: {extract_df.shape}')
   
    # Removing cases where the counter values were not logged (NaNs)
    print('TRANSFORM: Step 1 - Removing cases where the counter values were not logged (NaNs) or negative')
    mask_rm = extract_df['Value'].isnull() | (extract_df['Value'] < 0)
    fixed_counters_df = extract_df.loc[~mask_rm].copy()
    
    print(f'TRANSFORM: # removed records {sum(mask_rm)}')
    print(f'TRANSFORM: Shape after step: {fixed_counters_df.shape}')
    
    # fixing the counters
    ## calculating diffs between consecutive counters values
    group_by = ['TagName']
    fixed_counters_df = fixed_counters_df.sort_values(by=group_by+['DateTime'], ascending=True)
    fixed_counters_df.loc[:, qty_column_name] = fixed_counters_df.groupby(group_by)['Value'].diff().values
    fixed_counters_df = fixed_counters_df.assign(prev_value = fixed_counters_df['Value'] - fixed_counters_df[qty_column_name], 
                                                 next_value = fixed_counters_df.groupby(group_by)['Value'].shift(-1)
                                                )

    ## defining the counter reset
    mask_reset = (fixed_counters_df[qty_column_name] < 0) & ((fixed_counters_df['Value'] == 0)
             | ((fixed_counters_df['Value'] > 0) & (fixed_counters_df['prev_value'] != fixed_counters_df['next_value']))
             )
    fixed_counters_df = fixed_counters_df.assign(reset_flag = mask_reset)

    ## removing records, where the historian disconnect was observed
    print('TRANSFORM: Step 2 - Removing cases, where counter has dropped down in between two resets (historian outage issue)')
    mask_rm = (fixed_counters_df[qty_column_name] < 0) & ~fixed_counters_df['reset_flag']
    fixed_counters_df = fixed_counters_df.loc[~mask_rm].drop(columns=['prev_value', 'next_value', qty_column_name]).copy()

    print(f'TRANSFORM: # removed records {sum(mask_rm)}')
    print(f'TRANSFORM: Shape after step: {fixed_counters_df.shape}')

    ## re-calculating the counter differencies
    fixed_counters_df.loc[:, qty_column_name] = fixed_counters_df.groupby(group_by)['Value'].diff().values

    # removing records, where the counter was not changing
    print('TRANSFORM: Step 3 - Removing cases where the counter was not changing')
    mask_rm = (fixed_counters_df[qty_column_name] == 0) & (fixed_counters_df['Value'] != 0)
    fixed_counters_df = fixed_counters_df.loc[~mask_rm].copy()

    print(f'TRANSFORM: # removed records {sum(mask_rm)}')
    print(f'TRANSFORM: Shape after step: {fixed_counters_df.shape}')

    # removing records, where counter reports 0
    print('TRANSFORM: Step 4 - Removing cases where the counter reports zero')
    mask_rm = (fixed_counters_df['Value'] == 0)
    fixed_counters_df = fixed_counters_df.loc[~mask_rm].copy()

    print(f'TRANSFORM: # removed records {sum(mask_rm)}')
    print(f'TRANSFORM: Shape after step: {fixed_counters_df.shape}')

    # replacing NaNs in the beginning of the tag history or at the reset to the current value of the counter
    mask_replace =  fixed_counters_df[qty_column_name].isnull() | fixed_counters_df['reset_flag']
    fixed_counters_df.loc[mask_replace, qty_column_name] = fixed_counters_df.loc[mask_replace, 'Value']

    fixed_counters_df = fixed_counters_df.drop(columns=['reset_flag'])
    
    return fixed_counters_df


def add_site_line_tag(extract_df: pd.DataFrame, site: str, tag_name_column: str = 'TagName') -> pd.DataFrame:
    """Transforms datafame with fixed paking counters by adding 
      site, line, leg columns based on the TagName
      
      :param extract_df: dataframe containing fixed paking counters data
      
    """
    result_df = (extract_df
                    .assign(line = extract_df[tag_name_column].str.split(pat="_", n=2, expand=True)[0],
                            tag = extract_df[tag_name_column].str.split(pat="_", n=2, expand=True)[2],
                            site=site
                            )
                    .sort_values(by=[tag_name_column, 'DateTime'])
                    .drop(columns=[tag_name_column])   
                )
    print(f'TRANSFORM: Shape after addition of columns: {result_df.shape}')
    
    return result_df

def add_site_line_tag_new(extract_df: pd.DataFrame, site: str, tag_name_column: str = 'TagName') -> pd.DataFrame:
    """Transforms datafame with fixed paking counters by adding 
      site, line, leg columns based on the TagName
      
      :param extract_df: dataframe containing fixed paking counters data
      
    """
    result_dfn = (extract_df
                    .assign(line = extract_df[tag_name_column].str.split(pat="_", n=2, expand=True)[0],
                            site=site
                            )
                    .sort_values(by=[tag_name_column, 'DateTime'])
                    .drop(columns=[tag_name_column])   
                )
    print(f'TRANSFORM: Shape after addition of columns: {result_dfn.shape}')
    
    return result_dfn



def transform_counters_extract(extarct_df: pd.DataFrame, site_name: str, 
                               change_format: bool = True, counter_name: str ='rejects_qty',
                               local_tz: str = None) -> pd.DataFrame:
    """Transforms the raw historian extract with counters by:
        - puting table in standard format (wonderware)
        - removing bad quality records
        - transforming value column to numeric
        - fixing counters
        - adding columns with site, line and leg


    Args:
        extarct_df (pd.DataFrame): data frame with extract
        site_name (str): name of the site
        change_format (bool): whether the format of data frame should be 
                             changed to wonderware extract format
        counter_name (str): name of the counter column en resulting data frame
        lozal_tz (str): name of the timezone to convert time column

    Returns:
        pd.DataFrame: data frame with transformed extract
    """
    if change_format:
        extarct_df = put_to_wonderware_format(extarct_df, inplace=False)
    
    extarct_df = extarct_df.loc[extarct_df['Quality'] == 3].drop(columns=['Quality'])
    extarct_df = extarct_df.assign(Value = pd.to_numeric(extarct_df['Value'], errors='coerce'),
                                   DateTime = extarct_df['DateTime'].dt.tz_convert(local_tz).dt.tz_localize(None) 
                                   )
    extarct_df = fix_historian_counters(extarct_df, qty_column_name=counter_name)
    extarct_df = add_site_line_tag(extarct_df, site=site_name)

    return extarct_df

def upsert_data_to_datalab(data_frame: pd.DataFrame, db_conn, db_access_token,
                           target_table_nm: str, primary_key: list, 
                           query_create_template: str) -> None:
    """Upserts dataframe to the target table in DB based on provided primary key

    Args:
        data_frame (pd.DataFrame): dataframe to upsert with
        db_conn ([type]): DB connection object
        target_table_nm (str): name of the target table to upsert to in DB
        primary_key (list): primary key used for the upsert
        query_create_template (str): template of query creating the temporary table in DB to be used for the upsert.
                            Upsert is handled by DB, input data frame is inserted into the temp table first.
                            table name in the query should be parametrarized
    """
    # creating temp table in DB to insert dataframe
    table_tmp_nm = f'{target_table_nm}_TMP'
    query_create_tmp = query_create_template.format(table_tmp_nm, table_tmp_nm)
    execute_db_query(db_conn, query_create_tmp, db_access_token)

    # inserting data frame to the temp table
    print(f'Populating {table_tmp_nm} ...')
    insert_df_to_mssql_db(df=data_frame,
                          db_conn=db_conn,
                          db_table=table_tmp_nm,
                          access_token=db_access_token)
    
    # upserting
    upsert_query = get_merge_upd_ins_query(target_table_nm=target_table_nm,
                                           source_table_nm=table_tmp_nm,
                                           target_pk=primary_key,
                                           target_columns=data_frame.columns.tolist())
    print(f'Upserting the {target_table_nm} table with \n {upsert_query}')

    execute_db_query(db_conn, upsert_query, db_access_token)


def transpose_data_frame(data_frame: pd.DataFrame, rows: list, transpose_on: str) -> pd.DataFrame:
    """Transposes the table based on the specified settings

    Args:
        data_frame (pd.DataFrame): data frame to be transposed
        rows (list): list of columns, which will be forming unique row identifier
        transpose_on (str): name of the column to transpose on.

    Returns:
        pd.DataFrame: transposed data frame
    """

    index_cols = rows + [transpose_on]
    transposed_df = data_frame.set_index(index_cols).unstack(level=-1) 
    transposed_df.columns = [c[1] for c in transposed_df.columns.to_flat_index()]
    transposed_df = transposed_df.dropna(how='all').fillna(0).reset_index()

    return transposed_df









        
        

        

        