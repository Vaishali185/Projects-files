import pandas as pd
import numpy as np
import pyodbc as pyodbc

from sqlalchemy import create_engine
from urllib.parse import quote_plus
from typing import List, Tuple, Optional

from sud_utils import *

# etl related functions
# Extract

def extract_data_from_db(conn_string: str, select_query: str) -> pd.DataFrame:
    """Executes SELECT statement in DB and returns the resulting data in pandas data frame"""

    with pyodbc.connect(conn_string) as conn_db:
        try:
            cursor = conn_db.cursor()
            cursor.execute(select_query)
            data = cursor.fetchall()

            if len(data) == 0: # no data was extracted
                print('Connected! Executed statement returned 0 rows.\n')
                extract_df = pd.DataFrame()
            else:
                extract_df = pd.DataFrame().from_records(data)
                extract_df.columns = [column[0] for column in cursor.description]
                print(f'Connected! Shape of extracted data: {extract_df.shape}.\n')
        except Exception as e:
            extract_df = pd.DataFrame()
            cursor.rollback()
            print(f'Database Error: {e}')
        else:
            cursor.commit()
        finally:
            cursor.close()
    
    return extract_df


def execute_db_query(conn_string: str, query: str, access_token: bytes = None, **kwargs) -> None:
    """Executes query in DB"""

    with get_db_connection(conn_string, access_token) as conn_db:
        try:
            cursor = conn_db.cursor()
            cursor.execute(query)
        except Exception as e:
            cursor.rollback()
            print(f'Database Error: {e}')
        else:
            cursor.commit()
        finally:
            cursor.close()
            
    return None


# load

def insert_df_to_mssql_db(df: pd.DataFrame, db_conn: str, db_table: str, 
                          if_exists: str = 'append', index: bool = False, 
                          access_token: bytes = None, *args, **kwargs) -> None:
    """ Inserts data from pandas dataframe into MS-SQL db table
    
    :param df: (pandas.DataFrame) dataframe to write into the database
    :param db_conn: (string) database connection string
    :param db_table: (string) database table name to insert into
    :param if_exists: (string) {‘fail’, ‘replace’, ‘append’}, default ‘fail’
                            How to behave if the table already exists.
                                * fail: Raise a ValueError.
                                * replace: Drop the table before inserting new values.
                                * append: Insert new values to the existing table.
    :param index: (bool) default False. Write DataFrame index as a column. 
                          Uses index_label as the column name in the table.
    :param access_token: (string)
    
    """
    if if_exists not in ['fail','replace','append']:
        raise ValueError('if_exists expected to be one of the ["fail","replace","append"], but getting {}'.format(if_exists))
    else:
        conn_url = quote_plus(db_conn)
        try:
            if access_token is not None:
                engine = create_engine(f'mssql+pyodbc:///?odbc_connect={conn_url}', 
                                       connect_args={'attrs_before': {1256:access_token}})
            else:
                engine = create_engine(f'mssql+pyodbc:///?odbc_connect={conn_url}')
        except Exception as e:
            print(e)    
        
        try:
            df.to_sql(db_table, con=engine, if_exists=if_exists, index=index)
            engine.dispose()
        except Exception as e:
            print(e)
    
    return None


def add_dim_key_lookup(fact_df: pd.DataFrame, dim_df: pd.DataFrame, dim_id: str,
                       fact_on: List, dim_on: List, how: str ='inner') -> pd.DataFrame:
    """Adds demention key based on dimention table by lookup 
    
    :param fact_df: staging fact table
    :param dim_df: dimention table
    :param dim_id: name of id column in the dim_df
    :param fact_on: list of column to use for merge in fact table
    :param dim_on: list of column to use for merge in dimention table
    :param how: what kind of merge to be used. Defaults to 'inner'
    
    """
    
    dim_cols = dim_on + [dim_id]
    drop_cols = list(dict.fromkeys(fact_on + dim_on))
    
    result_df = (fact_df.merge(dim_df[dim_cols], 
                              left_on=fact_on, 
                              right_on=dim_on,
                              how=how)
                        .drop(columns=drop_cols)
                )
    
    return result_df


def add_dim_key_time(fact_df: pd.DataFrame, dim_df: pd.DataFrame, dim_id: str,
                     fact_dttm: str, dim_dttm: str, group_by: List) -> pd.DataFrame:
    
    """ Adds dimention key to the fact table based on the datetime column with forward fill.
    
    :param fact_df: staging fact table
    :param dim_df: dimention table
    :param dim_id: name of id column in the dimention table
    :param dim_dttm: datetime column in dimention table
    :param fact_dttm: datetime column in fact table
    :param group_by: columns to group by and fill. SHould be present in both tables
    
    """
    groupby_in_fact = [c for c in fact_df.columns if c in group_by]
    groupby_in_dim = [c for c in dim_df.columns if c in group_by]
    
    if len(groupby_in_fact) != len(groupby_in_dim):
        msg = f"These columns are expected to be in both fact and dim tables: {group_by}"
        return pd.DataFrame()
        raise ValueError(msg)
    else:
        if fact_dttm != dim_dttm:
            dim_df = dim_df.rename(columns={dim_dttm: fact_dttm})
        
        result_df = (fact_df.append(dim_df[group_by + [fact_dttm, dim_id]], sort=False, ignore_index=True)
                            .sort_values(by=group_by + [fact_dttm, dim_id])
                 
                 )
        result_df = result_df.assign(rm_flag = ~result_df[dim_id].isnull())
        result_df.loc[:, dim_id] = result_df.groupby(group_by)[dim_id].ffill()
        
        result_df = (result_df.loc[(result_df['rm_flag'] == False) 
                                 & (result_df[fact_dttm] >= fact_df[fact_dttm].min())]
                              .drop(columns=['rm_flag'])
                    )
        
        return result_df