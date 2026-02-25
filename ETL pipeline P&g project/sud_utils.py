# set of utils functions

import pandas as pd
import numpy as np
import pyodbc as pyodbc
import requests
import json
import struct
import pytz

from itertools import product
from sqlalchemy import create_engine
from datetime import datetime
from typing import List, Tuple, Optional



def get_mssql_conn_string(server_name: str, db_name: str, username: str = None, password: str = None, 
                          driver: str = 'ODBC Driver 17 for SQL Server', trusted: str = 'yes', 
                          with_token: bool = False, app_name: Optional[str] = None) -> str:
    """Generates connection string to the MS-SQL server 
    
    :param server_name: (string) MS-SQL server name
    :param db_name: (string) MS-SQL database name to connect to 
    :param username: (string) database user login
    :param password: (string) database user password 
    :param driver: (string) MS-SQL driver specification. Defaults to SQL Server
    :param trusted: (string) whether to use trusted connection. Defaults to yes
    :param with_token: (boolean) whether to generate connection string suitable to pass with token
    :param app_name: (string)  optional name of the connection. Defaults to None
    
    :returns conn_string: (string) connection string to the MS-SQL server to pass to 
                                    connect method of pyodbc
    
    
    """
    if with_token:
        conn_params = [server_name, db_name, driver]
        # checking parameters are of valid types
        if not all(isinstance(p, str) for p in conn_params):
            raise ValueError('Expecting strings for server_name, db_name, '
                             'driver, but getting: {}'.format([type(p) for p in conn_params]))
        else:
            conn_string = "Server={}; Database={}; Driver={};".format(*conn_params)
    else:
        if isinstance(trusted, str):
            if (trusted == 'no'):
                if (username is None) | (password is None):
                    raise ValueError('You have to provide username and password for not trusted connection')
                else:
                    conn_params = [server_name, db_name, username, password, driver]
                    # checking parameters are of valid types
                    if not all(isinstance(p, str) for p in conn_params):
                        raise ValueError('Expecting strings for server_name, db_name, '
                                         'username, password, driver, '
                                         'but getting: {}'.format([type(p) for p in conn_params]))
                    else:
                        conn_params.append(trusted)
                        conn_string = """ 
                            Server={}; Database={}; 
                            uid={}; pwd={}; 
                            Driver={}; Trusted_Connection={};
                            """.format(*conn_params)
            elif (trusted == 'yes'):
                if (username is None) | (password is None):
                    conn_params = [server_name, db_name, driver]
                    # checking parameters are of valid types
                    if not all(isinstance(p, str) for p in conn_params):
                        raise ValueError('Expecting strings for server_name, db_name, '
                                         'driver, but getting: {}'.format([type(p) for p in conn_params]))
                    else:
                        conn_params.append(trusted)
                        conn_string = """ 
                            Server={}; Database={}; 
                            Driver={}; Trusted_Connection={};
                            """.format(*conn_params)
                else:
                    conn_params = [server_name, db_name, username, password, driver]
                    # checking parameters are of valid types
                    if not all(isinstance(p, str) for p in conn_params):
                        raise ValueError('Expecting strings for server_name, db_name, '
                                         'username, password, driver, '
                                         'but getting: {}'.format([type(p) for p in conn_params]))
                    else:
                        conn_params.append(trusted)
                        conn_string = """ 
                            Server={}; Database={}; 
                            uid={}; pwd={}; 
                            Driver={}; Trusted_Connection={};
                            """.format(*conn_params)
            else:
                raise ValueError(f"trusted expected to be 'yes', 'no' or None, but getting {trusted}")
        else:
            raise ValueError(f'Expecting string for trusted, but getting {type(trusted)}')
            
        if app_name is not None:
            if isinstance(app_name, str):
                conn_string = conn_string + 'APP={}'.format(app_name)
            else:
                raise ValueError('Expecting string for app_name, but getting {}'.format(type(app_name)))
        else:
            pass
    
    return conn_string



def get_azure_sql_db_access_token(access_token_url: str) -> bytes:

    """ Getting access tocken to connect to the Azure SQL DB
    
    :param access_token_url: (str) access token URL

    """

    try:
        # request for access token
        response = requests.get(access_token_url, headers={'Content-Type': 'application/json', 'Metadata': 'true'})
        
        if response.status_code == requests.status_codes.codes.ok:
            # success...
            response_data = json.loads(response.text.encode('utf-8'))
            access_token = bytes(response_data['access_token'], 'utf-8')
            exp_token = b''

            for i in access_token:
                exp_token += bytes({i})
                exp_token += bytes(1)
            tokenstruct = struct.pack("=i", len(exp_token)) + exp_token

            return tokenstruct

        else:
            # error..
            print(f"Failure: Access tocken request failed with Status Code: {str(response.status_code)}")
            return None
    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)

        return None


def get_db_connection(conn_string: str, 
                      access_token: bytes = None, **kwargs) -> pyodbc.Connection:
    """ Returns puodbc connection based on the provided connection parameters.

    :param conn_string: (string) MS-SQL server connection string
    :param access_token_url: (bytes) access token URL
    
    :returns conn: (pyodbc.connection) connection to the MS-SQL

    """

    if access_token is not None:
        conn = pyodbc.connect(conn_string, attrs_before={1256: access_token}, **kwargs)        
    else:
        conn = pyodbc.connect(conn_string, **kwargs)

    
    return conn


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


# Historian-related utils

def get_tags_list(lines: List, sensors: List, sep: str = '_', topic: str = None) -> Tuple:
    """ Generates historian tags list based on list of lines and sensors
    
    :param lines: list of lines for which the tags are going to be generated
    :param sensors: list of sensors to be used for each line in tags generation
    :param sep: Default: '_'. Seperator to be used between line and sensor names
    """
    
    if not lines:
        raise ValueError(f'lines list is empty: {lines}')
    elif not sensors:
        raise ValueError(f'sensors list is empty: {sensors}')
    else:
        tags = ['_'.join(p) for p in product(lines, sensors)]
        if topic is not None:
            if isinstance(topic, str):
                tags = [f'{topic}.{tag}' for tag in tags]
            else:
                raise ValueError(f'topic should be string, but got: {type(topic)}')
        
        
    return tuple(tags)


def put_to_wonderware_format(data_frame: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Transforms proficy historian extract to be aligned 
    with wonderware extract

    Args:
        data_frame (pd.DataFrame): datafreme with Proficy historian extract
        inplace (bool): Whether to apply changes to original extract 

    Raises:
        ValueError: [description]

    Returns:
        pd.DataFrame: resulting dataframe
    """
    rename_dict = {'Tag':'TagName', 'Timestamp':'DateTime'}
    if inplace:

        data_frame = (data_frame.reset_index()
                                .rename(columns=rename_dict)
                                # .drop(columns=['Quality'])
                                )
        data_frame = data_frame.assign(TagName = data_frame['TagName'].str.split(pat=".", n=2, expand=True)[1])
        return data_frame
    else:
        df = data_frame.copy()
        df = (df.reset_index()
                .rename(columns=rename_dict)
                # .drop(columns=['Quality'])
            )
        df = df.assign(TagName = df['TagName'].str.split(pat=".", n=2, expand=True)[1])
        return df




# etl summary related utils

def get_last_run_df(table_nm: str, db_conn,
                    run_summary_table_nm: str = 'SUD_ETL_RUN_SUMMARY', 
                    status: str = 'success',
                    job_nm: str = None) -> pd.DataFrame:
    """Gets  dataframe which contains the record with the last results of etl run
       for specified table.
       
    :param table_nm: name of the table
    :param status: status of the run
    
    """
    if job_nm is None:
        get_last_run_query_tmpl = """
            select * 
            from {} as ers
            where ers.job_run_id in (
            select max(job_run_id) as run_id
            from {}
            where table_nm = '{}'
                and status = '{}'
            )
        """
        get_last_run_query = get_last_run_query_tmpl.format(run_summary_table_nm, 
                                                        run_summary_table_nm, 
                                                        table_nm, status)                                          
    else:
        get_last_run_query_tmpl = """
            select * 
            from {} as ers
            where ers.job_run_id in (
            select max(job_run_id) as run_id
            from {}
            where table_nm = '{}'
                and status = '{}'
                and job_nm = '{}'
            )
        """
        get_last_run_query = get_last_run_query_tmpl.format(run_summary_table_nm, 
                                                        run_summary_table_nm, 
                                                        table_nm, status, job_nm)   

    last_run_df = pd.read_sql(sql=get_last_run_query, con=db_conn)
    
    return last_run_df


def get_run_summary_entry(job_id: int, job_nm: str, table_nm: str,
                          start_time: datetime, end_time: datetime,
                          rows_processed: int, colid: str, coldttm: str, 
                          last_dttm_available: datetime,
                          last_id_available: int,
                          status: str = 'success', error_message: str = '') -> pd.DataFrame:
    """ Created dataframe with etl run summary entry 
    
    """
    
    job_run_summary_cols = ['job_id', 'job_nm', 'table_nm', 'start_time', 'end_time',
                            'rows_processed', 'status', 'error_message', 'colid', 'coldttm',
                            'last_dttm_available', 'last_id_available']
    job_run_summary_values = [job_id, job_nm, table_nm, start_time, end_time,
                              rows_processed, status, error_message, colid, coldttm,
                              last_dttm_available, last_id_available]
    
    job_summary_entry_df = pd.DataFrame(data=[job_run_summary_values], columns=job_run_summary_cols)
    
    return job_summary_entry_df

# Power BI related

def refresh_pbi_dataset(api_url: str, app_name: str) -> None:
    """ The Script refresh Power BI dataset using Logic Apps - trigger http 
    :param app_name: Logic Apps api url
    :param app_name: app name to refresh

    """
    print(f'Trying to trigger manual refresh for {app_name} app...')
    try:
        # Call data refresh by triggering http request..
        obj = '{"AppName": "'+app_name+'"}'
        header = {"content-type": "application/json"}
        response = requests.post(api_url, data=obj, headers=header)
        if response.status_code == 200 or response.status_code == 202:
            # success...
            print("Success: Dataset Refresh Triggered")
        else:
            # error..
            print(f'Failure: Dataset Refresh Triggered Failed with Status Code: {str(response.status_code)}')
            #log(response_error, ERROR, APP_LOG)
    except Exception as ex:
        template = "An exception of type {0} occurred in mes_refresh_dataset.py(mes_trigger_dataset_refresh). Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        

# DATETIME related functions

def utc_to_local(utc_dt: datetime, local_tz: str) -> datetime:
    """Convert UTC datetime to datetime in specified tz"""
    
    tz = pytz.timezone(local_tz)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(tz)
    
    return tz.normalize(local_dt)

def to_local_timestr(utc_dt: datetime, local_tz: str, formt: str = '%Y%m%d %H:%M:%S') -> str:
    """Convert UTC datetime to datetime string in specified tz with specified format.
       Default format is %Y%m%d %H:%M:%S - corresponding to Wonderware Historian"""
    
    return utc_to_local(utc_dt, local_tz).strftime(formt)
    
def get_tz_diff(date: datetime, tz_from: str, tz_to: str) -> float:
    """Returns the difference in hours between tz_from and tz_to for a given date.
    """
    
    tz1 = pytz.timezone(tz_from)
    tz2 = pytz.timezone(tz_to)
    date = pd.to_datetime(date)

    return (tz1.localize(date) - tz2.localize(date).astimezone(tz1)).total_seconds()/3600 


def get_tool_hist_tags_table(tags: List, tool_nm: str, site_nm: str):
    """Generates table, which used to populate the SUD_PBI_TOOL_TAGS table
    :param tags: list of tags in standard historian format LXX_TTT
    :param tool_nm: name of the tool
    :param site_nm: name of the site
    """
    
    if tags:
        tool_tags_df = pd.DataFrame({'TagName':tags})
        tool_tags_df = tool_tags_df.assign(ToolName=tool_nm,
                                            Site = site_nm,
                                            Line = tool_tags_df['TagName'].str.split('_', n=1, expand=True)[0],
                                            LastModifiedDate = datetime.now()
                                          )
    else:
        tool_tags_df = None
        raise ValueError('List of tags is empty')
    
    return tool_tags_df


def generate_taglist_table(tag_list: List, lines_list: List, site_nm: str, histo_topic: str, tool_nm: str) -> pd.DataFrame:
    """ Function to generate a tag list table to be populated into SUD_PBI_TOOL_TAG in Datalab DB
    
    Args:
        tag_list (list): List of tag names (i.e. TOPIC.LXXX_Tag_Name)
        lines_list (list): List of line names in historian notation (i.e. LXXX)
        site_nm (str): Site name
        histo_topic (str): Site specific historian topic
        tool_nm (str): names of the dashboard/tool

    Returns:
        pd.DataFrame: dataframe containing tagg used in the specific tool in format matching the SUD_PBI_TOOL_TAG table 
    
    """
    all_tag_list = get_tags_list(lines=lines_list, sensors=tag_list ,sep='_', topic=histo_topic)
    tag_df = pd.DataFrame({'TagName':all_tag_list})
    tag_df = tag_df.assign(Site=site_nm,
                           ToolName=tool_nm,
                           LastModifiedDate=datetime.now(),
                           Line=(tag_df['TagName'].str.split('.',n=1,expand=True)[1]
                                                .str.split('_',n=1,expand=True)[0]
                                                .str[:-1]),
                           Technology='BAG'
                        )
    tag_df = tag_df[['ToolName','Site','Line','TagName','LastModifiedDate','Technology']]

    return tag_df
    
    