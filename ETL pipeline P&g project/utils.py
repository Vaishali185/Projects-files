import queries
import pandas as pd
import numpy as np
import pyodbc as pyodbc
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from typing import List, Tuple, Optional
from datetime import datetime

from commons import *

# EXTRACT functions

def get_histn_extract(conn: str, tags: Tuple, 
                      start_dt: datetime, end_dt: datetime, 
                      timezone: str, extract_type: str = 'Delta', resolution: Optional[int] = None) -> pd.DataFrame:
    """ Perform extraction from the wwHistorian database based on the provided imputs
    :param conn: (string) MS-SQL connection string to pass to pyodbc
    :param tags: (list-like) list of tags to extract
    :param start_dt: (datetime) start timestamp
    :param end_dt: (datetime) end timestamp
    :param timezone: (string) local timezone for the site
    :param extract_type:(string) type of extract to use for historian
    :param resolution: (int) in case of cyclic extraction this parameter provides resolution
    
    :returns pandas.DataFrame() containing the data extract
    """
    # input checks
    extract_types = ['Delta', 'Cyclic']
    
    if extract_type in extract_types:
        if extract_type == 'Cyclic':
            if resolution is None:
                raise ValueError('Please provide resolution for cyclic extraction')
            elif not isinstance(resolution, int):
                raise ValueError('Expectiong integer for resolution, but getting {}'.format(type(resolution)))
            else:
                pass
    else:
        raise ValueError('extract_type expected to be one of the following: {}'.format(extract_types))
        
    if not isinstance(tags, (list, tuple)):
        raise ValueError('tags expected to be a list, but getting {}'.format(type(tags)))
    else:
        pass
    
    # constructing the query
    import queries
    
    start_dt_htn = to_local_timestr(start_dt, timezone)
    end_dt_htn = to_local_timestr(end_dt, timezone)
    
    if extract_type == 'Delta':
        query = queries.HISTN_SELECT_DELTA.format(tags, start_dt_htn, end_dt_htn)
    else:
        query = queries.HISTN_SELECT_CYCLYC.format(tags, start_dt_htn, end_dt_htn, resolution)
    
    # executing the query 
    conn_hstn = pyodbc.connect(conn)
    with conn_hstn:
        cursor_hstn = conn_hstn.cursor()
        cursor_hstn.execute(query)
    
        data = cursor_hstn.fetchall()
        df = pd.DataFrame().from_records(data)
        if df.shape[0] <= 0:
            df = pd.DataFrame()
        else:
            df.columns = [column[0] for column in cursor_hstn.description]

        cursor_hstn.close()
    
    return df 

# TRANSFORM functions

    
    
# LOAD related functions

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
    

def delete_latest_data_mssql(db_conn: str, db_table: str, dttm_column: str, 
                            last_dttm: datetime, operator: str ='ge', dttm_format: str = '%m/%d/%Y %H:%M:%S') -> None:
    """Removes rows from the table for with datetimes later then specified timestamp
    
    :param db_conn: (string) connection string to pass to pyodbc
    :param db_table: (string) name of the database table
    :param dttm_column: (string) name of the datetime column to filter on
    :param last_dttm: (datetime.datetime) this timestamp will be used to filter out the database table
    :param operator: (string) (ge, g, le, l, e, ne), default: ge i.e. >=. 
                               Comparison operator to use in the where statement
    """
    allow_operators = ['ge', 'g', 'le', 'l', 'e', 'ne']
    sql_operators = ['>=', '>', '<=', '<', '=', '!=']
    operators_map = dict(zip(allow_operators, sql_operators))
    
    if operator not in allow_operators:
        raise ValueError('operator expected to be one of the following : {}, but getting {}'.format(allow_operators, operator))
    else:
        sql_operator = operators_map[operator]
    
    import queries
    
    conn_db = pyodbc.connect(db_conn, autocommit=True)
    
    # preparing query
    dttm = last_dttm.strftime(dttm_format)
    query = queries.DB_DELETE_ROWS_WHERE_ONE_COLUMN.format(db_table, dttm_column, sql_operator, f"'{dttm}'")
    
    try:
        with conn_db:
            cursor = conn_db.cursor()
            cursor.execute(query)
            cursor.close()
    except Exception as e:
        print(e)
        
    return None

def upsert_mssql_db_table_on_dttm(df: pd.DataFrame, db_conn: str, db_table: str, dttm_column: str,  
                                  dttm_format: str = '%m/%d/%Y %H:%M:%S', if_exists: str = 'append', 
                                  index: bool = False, *args, **kwargs) -> None:
    """Performs upsert of the database tabla from pandas dataframe, based on the datetime column.
       At the first step all the rows for which values of datatime column are greater then the first datetime
       from the dataframe are deleted from db table. After that the dataframe values are inserted into the db table.
       
       :param df: (pandas.DataFrame) dataframe to be inserted into into the database
       :param db_conn: (string) connection string to connect ot the database containing target table
       :param db_table: (string) name of the target table to insert into
       :param dttm_column: (string) names of the timestamp column. To be used in the upsert step
       :param dttm_format: (string) format of the datetime to be used in the target table
       :param if_exists: (string) {‘fail’, ‘replace’, ‘append’}, default ‘append’
                            How to behave if the table already exists.
                                * fail: Raise a ValueError.
                                * replace: Drop the table before inserting new values.
                                * append: Insert new values to the existing table.
       :param index: (bool) default False. Write DataFrame index as a column. 
                          Uses index_label as the column name in the table.
    """
    
    # get latest timestamp from df
    start_dttm = df[dttm_column].min()
    
    # remove all rows from the target dbtable for which dttm_column >= last_dttm
    try:
        delete_latest_data_mssql(db_conn, db_table, dttm_column, start_dttm, dttm_format='%m/%d/%Y %H:%M:%S')
    except Exception as e:
        print(e)
    
    # insert dataframe into target db table   
    try:
        insert_df_to_mssql_db(df, db_conn, db_table, if_exists, index, *args, **kwargs)
    except Exception as e:
        print(e)
        
    return None
    

def get_merge_upd_ins_query(target_table_nm: str, source_table_nm: str,
                            target_pk: List, target_columns: List,
                            source_pk: List = None, source_columns: List = None) -> str:
    """ Generates simple upsert sql query based on MERGE statement.
    
    :param target_table_nm: name of the target table into which the upsert is performed
    :param source_table_nm: name of the source table from which the upsert is performed
    :param target_pk: list of columns forming the primary key in target table
    :param target_columns: list of columns to be updated/inserted in target table 
    :param source_pk: list of columns forming the primary key in source table
    :param source_columns: list of source table columns which will be used for update/insert
    
    """
    
    if source_pk is None:
        print(f"Set of primary keys for source table wasn't provided. Asumming same as in target table: {target_pk}")
        source_pk = target_pk
    elif len(target_pk) != len(source_pk):
        raise ValueError('Number of columns in target_pk and source_pk should be the same.')
    elif len(target_pk) == 0:
        raise ValueError('No primary key columns specified for target table')
    else:
        pass
        
        
    if source_columns is None:
        print(f"Set of source table columns to upsert wasn't provided. Asumming same as in target table: {target_columns}")
        source_columns = target_columns
    elif len(target_columns) != len(source_columns):
        raise ValueError('Number of columns in target_columns and source_columns should be the same.')
    elif len(target_columns) == 0:
        raise ValueError('No update/insert columns specified for target table')
    else:
        pass
    
    target_columns = [c for c in target_columns if c not in target_pk]
    source_columns = [c for c in source_columns if c not in source_pk]
    
    merge_on = 'AND '.join([f'target.[{t}] = source.[{s}] ' for t,s in zip(target_pk, source_pk)])
    update_set = ', '.join([f'target.[{t}] = source.[{s}]' for t,s in zip(target_columns, source_columns)])
    insert_into_columns = ', '.join([f'[{c}]' for c in target_pk + target_columns])
    insert_columns = ', '.join([f'source.[{c}]' for c in source_pk + source_columns])
    
    import queries
    query = queries.DB_UPSERT_MERGE.format(target_table_nm, source_table_nm, merge_on,
                                           update_set, insert_into_columns, insert_columns)
    
    return query