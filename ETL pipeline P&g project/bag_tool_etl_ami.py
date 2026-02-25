#
#
#

import pandas as pd
import pyodbc as pyodbc

import queries
import logging

import sys
import os
import logging
import sys
import argparse
import pytz
from os.path import join, split 

from datetime import datetime, timedelta
from tomlkit import parse, dumps, loads
from typing import List, Tuple

import historian

from sud_tools import *
from sud_utils import *
from tool_utils import *

# parsing comand line arguments
# TODO: add parameter for site code
parser = argparse.ArgumentParser(description='ETL script for Bag tool')
parser.add_argument('cfg_path', help='Full path to ETL config file')
# parser.add_argument('-s', '--site', 
#                     choices = ['ami', 'alx', 'lim', 'url', 'tak'], 
#                     default='ami',
#                     nargs=1,
#                     required=False,
#                     help='SUD site code to run etl for')

site_cd = 'ami'
args = parser.parse_args()
cfg_file_path = args.cfg_path

# checking if config file exists
if not os.path.isfile(cfg_file_path):
    raise ValueError(f'No config file was found at: {cfg_file_path}')
else:
    # reading in the config file
    cfg = loads(open(cfg_file_path).read())

job_id = 10*cfg['job']['id']
site_job_id_offset = cfg['sites'][site_cd]['job_id_offset']
job_id = job_id + site_job_id_offset

job_nm = ': '.join((cfg['job']['name'], site_cd))
job_summary_table_nm = cfg['job']['job_summary_table_nm']
job_start_dttm = datetime.now()

logs_path = cfg['job']['path_to_logs']

# cheking if the logs folder exist
if not os.path.exists(logs_path):
    print(f'INFO: Didn not find logs folder at {logs_path}. Creating it...')
    os.makedirs(os.path.join(logs_path))

# creating log file to write to 

log_create_dttm = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
log_file_nm = f"{cfg['job']['name']}_{job_id}_{log_create_dttm}.log"
log_file_path = os.path.join(logs_path, log_file_nm)

log_file = open(log_file_path, 'w+')
log_file.close()

print(f'Redirecting logging and stdout to {log_file_path}')

# setting up logging to a file
logging.basicConfig(format='[%(asctime)s] : %(levelname)s : %(funcName)s : %(lineno)d - %(message)s',
                    level=logging.INFO,
                    filename=log_file_path)
sys.stdout = open(log_file_path, 'a')

# reading the tag list file
tags_list_path = cfg['job']['path_to_tag_list']
print(f'Path to tag list: {tags_list_path}')

# cheking if the logs folder exist
if not os.path.exists(logs_path):
    print(f'INFO: Didn not find logs folder at {logs_path}.')
    raise ValueError('Was not able to file tags list. Aborting...')
else:
    tags_list_df = pd.read_excel(tags_list_path)
    tags_list_df = tags_list_df.loc[tags_list_df['Enabled'] == 1].copy()
    print(f'Shape of the tags list table: {tags_list_df.shape}')


if __name__ == '__main__':

    logging.info(f'Starting ETL for: {site_cd.upper()}')

    # datalab integration
    datalab_cfg = cfg['datalab_db'].copy()

    # setting up run configuration with parameters from cfg file
    datalab_db_conn = get_mssql_conn_string(**datalab_cfg['connection'])
    datalab_db_access_url = r"{}".format(datalab_cfg['access_token_url']['access_token_url'])
    datalab_db_access_token = get_azure_sql_db_access_token(datalab_db_access_url)

    pbi_refresh_url = cfg['powerbi']['api_url']
    pbi_refresh_app = cfg['powerbi']['app_name']

    pack_rejects_table_nm = datalab_cfg['output_tables']['pack_rejects_table_nm']
    pack_production_table_nm = datalab_cfg['output_tables']['pack_production_table_nm']
    tag_list_table_nm = datalab_cfg['output_tables']['tag_table']

    pack_rejects_agg_freq = datalab_cfg['output_tables']['pack_rejects_agg_freq']
    pack_production_agg_freq = datalab_cfg['output_tables']['pack_prod_agg_freq']

    site_cfg = cfg['sites'][site_cd].copy()

    site_name = site_cfg['site_name']
    site_enabled = site_cfg['enabled']
    site_agile_enabled = site_cfg['agile_enabled']
    site_project_enabled = site_cfg['project_enabled']
    site_history_buffer_days = site_cfg['history_buffer_days']

    site_servers = site_cfg['servers'].copy()
    site_lines = site_cfg['lines']['lines']
    site_lines_dim = site_cfg['lines']['lines_dim']
    site_lines_agile = site_cfg['lines']['lines_agile']

    site_historian_tags = site_cfg['tags'].copy()

    site_tz = site_servers['timezone']
    site_dttm_format = site_servers['dttm_format']
    site_historian_source = site_servers['historian']['source']
    site_history_depth = site_servers['historian']['history_depth_days']
    site_days_to_retake = site_history_depth + site_history_buffer_days

    site_tags = tags_list_df.loc[tags_list_df['Site'] == site_name, 'Historian tag'].unique()
    site_tags = ['_'.join(t.split('_')[1:]) for t in site_tags]

    line_maping_dict = dict(zip(site_lines, site_lines_dim))

    # populating the tag list table in Datalab DB
    tag_list_df = generate_taglist_table(site_tags, site_lines, site_name, site_servers['historian']['proficy']['topic'], pbi_refresh_app)
    
    query_to_execute = queries.TRUNCATE_TAG_LIST_TABLE.format(tag_list_table_nm, site_name, pbi_refresh_app)
    execute_db_query(datalab_db_conn, query_to_execute, datalab_db_access_token)
    
    insert_df_to_mssql_db(df=tag_list_df, 
                          db_conn=datalab_db_conn,
                          db_table=tag_list_table_nm, 
                          access_token=datalab_db_access_token,
                          if_exists='append'
                        )

    # Reading in the SUD Site and Line dimention tables

    query_to_execute = "SELECT * FROM SUD_SITES"
    sites_dim_df = pd.read_sql(sql=query_to_execute, con=get_db_connection(datalab_db_conn, datalab_db_access_token))
    logging.info(f'SUD sites dimention table loaded, shape: {sites_dim_df.shape}')

    query_to_execute = "SELECT * FROM SUD_LINES"
    lines_dim_df = pd.read_sql(sql=query_to_execute, con=get_db_connection(datalab_db_conn, datalab_db_access_token))
    logging.info(f'SUD lines dimention table loaded, shape: {lines_dim_df.shape}')

    job_step_start_dttm = datetime.now()

    if site_enabled == True:
        if site_historian_source == 'proficy':
            # set up

            non_agile_tags = [t for t in site_tags if t not in site_historian_tags['tags_agile'] + site_historian_tags['tags_projects']]
            agile_tags = [t for t in site_tags if t in site_historian_tags['tags_agile']]
            project_tags = [t for t in site_tags if t in site_historian_tags['tags_projects']]

            site_line_tags = get_tags_list(lines=site_lines, sensors=non_agile_tags, sep='_', 
                                           topic=site_servers['historian']['proficy']['topic'])
            if agile_tags:
                site_agile_tags = get_tags_list(lines=site_lines_agile, sensors=agile_tags, sep='_', 
                                                topic=site_servers['historian']['proficy']['topic'])
            else: 
                site_agile_tags = []
            
            if project_tags:
                site_project_tags = get_tags_list(lines=site_lines, sensors=project_tags, sep='_', 
                                                 topic=site_servers['historian']['proficy']['topic'])
            else: 
                site_project_tags = []

            index_tags = [t for t in site_line_tags if '_'.join(t.split('_')[1:]) in site_historian_tags['tags_index']]
            tags_total_prod = [t for t in site_line_tags if t.endswith('Good_Bag_Total')]
            tags_total_rejects = [t for t in site_line_tags if not t.endswith('Good_Bag_Total') and 'total' in t.lower()]
            tags_rejects = [t for t in site_line_tags if (t not in index_tags + tags_total_prod + tags_total_rejects) 
                                                        or 'changeover' in t.lower()]

            historian.use_context('REST', 
                    client_id='historian_public_rest_api', 
                    client_password='publicapisecret',
                    app_id='sudanalytics.im', 
                    app_password='phoenix2021SUD', # change to environmental variable
                    port=site_servers['historian']['proficy']['port'],
                    verify_certificate=True)
            
            # define extract window
            last_run_df = get_last_run_df(table_nm=pack_rejects_table_nm,
                        db_conn=get_db_connection(datalab_db_conn, datalab_db_access_token),
                        job_nm=job_nm)

            if not last_run_df.empty:
                last_available_dttm = last_run_df['last_dttm_available'][0]
                last_available_id = last_run_df['last_id_available'][0]
            else:
                last_available_dttm = None
                last_available_id = None

            start_dttm = datetime.now() if last_available_dttm is None else last_available_dttm
            start_dttm = start_dttm - timedelta(days=site_days_to_retake)
            start_time = start_dttm
            end_time = datetime.now()
            logging.info(f'Extracting data between: {start_time} and {end_time}')

            try:
                # extract
                ## extracting machine status tags
                machine_status_extract_df = historian.get_tag_values(
                    site_servers['historian']['proficy']['server_name'],
                    start_time=start_time - timedelta(days=1),
                    end_time=end_time,
                    filter_name=index_tags)
                logging.info(f'Shape of machine speed extract: {machine_status_extract_df.shape}')
                
                machine_status_df = put_to_wonderware_format(machine_status_extract_df, inplace=False)
                machine_status_df = machine_status_df.assign(DateTime = machine_status_df['DateTime'].dt.tz_convert(site_tz).dt.tz_localize(None))
                machine_status_df = add_site_line_tag(machine_status_df, site=site_name)
                machine_status_df = machine_status_df.drop(columns=['tag'])

                ## extract rejects
                mespack_rejects_extract_df = historian.get_tag_values(
                    site_servers['historian']['proficy']['server_name'],
                    start_time=start_time,
                    end_time=end_time,
                    filter_name=tags_rejects)
                logging.info(f'Shape of bag rejects extract: {mespack_rejects_extract_df.shape}')

                ## extract line state
                extract_dttm = datetime.strftime(start_time.date(), '%Y%m%d %H:%M:%S') 
                query_to_execute = queries.EXTRACT_LINE_STATE.format(extract_dttm, tuple(site_lines_dim))

                line_state_map_df = pd.read_sql(sql=query_to_execute, con=get_db_connection(datalab_db_conn, datalab_db_access_token))
                logging.info(f'Line state was extracted, shape: {line_state_map_df.shape}')

                index_cols = ['line_state_id', 'line', 'site']
                line_state_map_df = (line_state_map_df
                                    .set_index(index_cols)
                                    .stack()
                                    .to_frame('DateTime')
                                    .reset_index()
                                    .drop(columns=[f'level_{len(index_cols)}'])
                                    )
                # transform
                mespack_rejects_df = transform_counters_extract(mespack_rejects_extract_df, site_name=site_name, local_tz=site_tz)
                ## add machine speed
                mespack_rejects_df = add_dim_key_time(fact_df=mespack_rejects_df, 
                                            dim_df=machine_status_df.rename(columns={'Value':'Machine_Speed'}), 
                                            dim_id='Machine_Speed', fact_dttm='DateTime', dim_dttm='DateTime', 
                                            group_by=['site', 'line'])

                ## add agile flag if enabled
                if site_agile_enabled == True:
                    logging.info(f'Agile is enabled for the site.')
                    logging.info(f'Extracting data between: {start_time} and {end_time}')
                    agile_extract_df = historian.get_tag_values(site_servers['historian']['proficy']['server_name'], 
                                                                        start_time=start_time - timedelta(days=1),
                                                                        end_time=end_time,
                                                                        filter_name=site_agile_tags)
                    logging.info(f'Shape of agile flag extract: {agile_extract_df.shape}')

                    if agile_extract_df.shape[0] > 0:
                        agile_status_df = put_to_wonderware_format(agile_extract_df, inplace=False)
                        agile_status_df = agile_status_df.assign(DateTime = agile_status_df['DateTime'].dt.tz_convert(site_tz).dt.tz_localize(None),
                                                                Value = agile_status_df['Value'].astype(int))
                        agile_status_df = add_site_line_tag(agile_status_df, site=site_name)
                        agile_status_df = agile_status_df.drop(columns=['tag'])

                        agile_lines_mask = lines_dim_df['line_historian'].isin(site_lines_agile) & ~lines_dim_df['is_vec']
                        agile_legs_df = lines_dim_df.loc[agile_lines_mask, ['line_historian', 'leg']]
                        agile_status_df = agile_status_df.merge(agile_legs_df, left_on='line', right_on='line_historian', how='inner')
                        agile_status_df = (agile_status_df
                                .assign(line = agile_status_df['line'].str.cat(agile_status_df['leg']))
                                .drop(columns=['leg', 'line_historian']))

                        ## add agile flag
                        mespack_rejects_df = add_dim_key_time(fact_df=mespack_rejects_df, 
                                                    dim_df=agile_status_df.rename(columns={'Value':'Agile_Flag'}), 
                                                    dim_id='Agile_Flag', fact_dttm='DateTime', dim_dttm='DateTime', 
                                                    group_by=['site', 'line'])
                        mespack_rejects_df = mespack_rejects_df.assign(Agile_Flag = mespack_rejects_df['Agile_Flag'].fillna(0))
                        is_agile = True
                    else:
                        mespack_rejects_df = mespack_rejects_df.assign(Agile_Flag = 0)
                        is_agile = False
                else:
                    mespack_rejects_df = mespack_rejects_df.assign(Agile_Flag = 0)
                    is_agile = False

                ## add project flag if enabled
                if site_project_enabled == True:
                    logging.info(f'Project is enabled for the site.')
                    logging.info(f'Extracting data between: {start_time} and {end_time}')
                    project_extract_df = historian.get_tag_values(site_servers['historian']['proficy']['server_name'], 
                                                                  start_time=start_time - timedelta(days=1),
                                                                  end_time=end_time,
                                                                  filter_name=site_project_tags)
                    logging.info(f'Shape of agile flag extract: {project_extract_df.shape}')

                    if project_extract_df.shape[0] > 0:
                        project_status_df = put_to_wonderware_format(project_extract_df, inplace=False)
                        project_status_df = project_status_df.assign(DateTime = project_status_df['DateTime'].dt.tz_convert(site_tz).dt.tz_localize(None),
                                                                    Value = project_status_df['Value'].astype(int))
                        project_status_df = add_site_line_tag(project_status_df, site=site_name)
                        project_status_df = project_status_df.drop(columns=['tag'])

                        ## add project tag
                        mespack_rejects_df = add_dim_key_time(fact_df=mespack_rejects_df, 
                                                    dim_df=project_status_df.rename(columns={'Value':'Project_Flag'}), 
                                                    dim_id='Project_Flag', fact_dttm='DateTime', dim_dttm='DateTime', 
                                                    group_by=['site', 'line'])
                        mespack_rejects_df = mespack_rejects_df.assign(Project_Flag = mespack_rejects_df['Project_Flag'].fillna(0))
                        is_project = True
                    else:
                        mespack_rejects_df = mespack_rejects_df.assign(Project_Flag = 0)
                        is_project = False
                else:
                    mespack_rejects_df = mespack_rejects_df.assign(Project_Flag = 0)
                    is_project = False

                ## add line state
                mespack_rejects_df = add_dim_key_time(fact_df=mespack_rejects_df.assign(line = mespack_rejects_df['line'].map(line_maping_dict)), 
                                            dim_df=line_state_map_df, 
                                            dim_id='line_state_id', fact_dttm='DateTime', dim_dttm='DateTime', 
                                            group_by=['site', 'line'])

                ## aggregating
                mespack_rejects_df = mespack_rejects_df.assign(DateTimeMin = mespack_rejects_df['DateTime'].dt.floor(pack_rejects_agg_freq))

                groupby_cols = ['site', 'line', 'DateTimeMin', 'tag', 
                                'Machine_Speed', 'Agile_Flag', 'Project_Flag', 'line_state_id']
                agg_dict = {'rejects_qty':'sum', 'DateTime':'min'}

                mespack_rejects_agg_df = mespack_rejects_df.groupby(groupby_cols).agg(agg_dict).reset_index()

                ## adding line_id and site_id
                rename_dict = {'DateTimeMin':'datetime', 'tag':'reject_type', 
                               'Machine_Speed':'machine_speed', 'DateTime':'start_time', 
                               'Agile_Flag':'agile_flag', 'Project_Flag':'project_flag'}
                mespack_rejects_agg_df = (mespack_rejects_agg_df
                                    .merge(lines_dim_df[['line_id', 'line']], on = 'line', how='inner')
                                    .merge(sites_dim_df[['site_id', 'site']], on = 'site', how='inner')
                                    .drop(columns=['site', 'line'])
                                    .rename(columns=rename_dict)
                                )
                # loading
                truncate_date = start_time + timedelta(days=site_history_buffer_days)
                site_sud_id = sites_dim_df.loc[sites_dim_df['site'] == site_name, 'site_id'].iloc[0]
                mespack_rejects_agg_df = mespack_rejects_agg_df.loc[mespack_rejects_agg_df['datetime'] >= truncate_date]

                truncate_query = queries.TRUNCATE_REJECTS_TABLE.format(pack_rejects_table_nm, site_sud_id, truncate_date)
                execute_db_query(conn_string=datalab_db_conn, query=truncate_query, access_token=datalab_db_access_token)
                insert_df_to_mssql_db(df=mespack_rejects_agg_df,
                                    db_conn=datalab_db_conn,
                                    db_table=pack_rejects_table_nm,
                                    access_token=datalab_db_access_token)

                # updating etl status table 
                extract_overview_df = (mespack_rejects_agg_df.groupby(['site_id', 'line_id'])
                                        .agg({'datetime':'max', 'line_state_id':'count'})
                                        .reset_index()
                                        .rename(columns={'line_state_id':'n_records', 'datetime':'start_time'}))
                new_last_available_dttm = extract_overview_df['start_time'].min()

                job_summary_record = get_run_summary_entry(job_id, job_nm, pack_rejects_table_nm,
                                                            job_step_start_dttm, datetime.now(),
                                                            extract_overview_df['n_records'].sum(),
                                                            np.nan, 'start_time', 
                                                            new_last_available_dttm, np.nan)
                insert_df_to_mssql_db(df=job_summary_record,
                          db_conn=datalab_db_conn,
                          db_table=job_summary_table_nm,
                          access_token=datalab_db_access_token)

            except Exception as e:
                extract_overview_df = pd.DataFrame()
                agile_extract_df = pd.DataFrame()
                project_extract_df = pd.DataFrame()

                job_summary_record = get_run_summary_entry(job_id, job_nm, pack_rejects_table_nm,
                                                            job_step_start_dttm, datetime.now(),
                                                            0, np.nan, 'start_time', 
                                                            np.nan, np.nan, status='failed',
                                                            error_message=str(e))

                insert_df_to_mssql_db(df=job_summary_record,
                          db_conn=datalab_db_conn,
                          db_table=job_summary_table_nm,
                          access_token=datalab_db_access_token)


            # define extract window
            last_run_df = get_last_run_df(table_nm=pack_production_table_nm,
                        db_conn=get_db_connection(datalab_db_conn, datalab_db_access_token),
                        job_nm=job_nm)

            if not last_run_df.empty:
                last_available_dttm = last_run_df['last_dttm_available'][0]
                last_available_id = last_run_df['last_id_available'][0]
            else:
                last_available_dttm = None
                last_available_id = None

            start_dttm = datetime.now() if last_available_dttm is None else last_available_dttm
            start_dttm = start_dttm - timedelta(days=site_days_to_retake)
            start_time = start_dttm
            end_time = datetime.now()
            logging.info(f'Extracting data between: {start_time} and {end_time}')

            try:
                # extract totals
                ## extract total production/rejects
                mespack_total_rejects_extract_df = historian.get_tag_values(
                    site_servers['historian']['proficy']['server_name'],
                    start_time=start_time,
                    end_time=end_time,
                    filter_name=tags_total_prod + tags_total_rejects)
                logging.info(f'Shape of bag production and total rejects extract: {mespack_total_rejects_extract_df.shape}')
                
                # transform
                mespack_total_rejects_df = transform_counters_extract(mespack_total_rejects_extract_df, 
                                                    site_name=site_name, counter_name='qty', local_tz=site_tz)
                ## add machine speed
                mespack_total_rejects_df = add_dim_key_time(fact_df=mespack_total_rejects_df, 
                                            dim_df=machine_status_df.rename(columns={'Value':'Machine_Speed'}), 
                                            dim_id='Machine_Speed', fact_dttm='DateTime', dim_dttm='DateTime', 
                                            group_by=['site', 'line'])

                ## add agile flag if enabled
                if is_agile:
                    mespack_total_rejects_df = add_dim_key_time(fact_df=mespack_total_rejects_df, 
                                                dim_df=agile_status_df.rename(columns={'Value':'Agile_Flag'}), 
                                                dim_id='Agile_Flag', fact_dttm='DateTime', dim_dttm='DateTime', 
                                                group_by=['site', 'line'])
                    mespack_total_rejects_df = mespack_total_rejects_df.assign(Agile_Flag = mespack_total_rejects_df['Agile_Flag'].fillna(0))
                else:
                    mespack_total_rejects_df = mespack_total_rejects_df.assign(Agile_Flag = 0)

                ## add projects flag if enabled
                if is_project:
                    mespack_total_rejects_df = add_dim_key_time(fact_df=mespack_total_rejects_df, 
                                                dim_df=project_status_df.rename(columns={'Value':'Project_Flag'}), 
                                                dim_id='Project_Flag', fact_dttm='DateTime', dim_dttm='DateTime', 
                                                group_by=['site', 'line'])
                    mespack_total_rejects_df = mespack_total_rejects_df.assign(Project_Flag = mespack_total_rejects_df['Project_Flag'].fillna(0))
                else:
                    mespack_total_rejects_df = mespack_total_rejects_df.assign(Project_Flag = 0)

                ## adding line state
                mespack_total_rejects_df = add_dim_key_time(
                    fact_df=mespack_total_rejects_df.assign(line = mespack_total_rejects_df['line'].map(line_maping_dict)), 
                    dim_df=line_state_map_df, 
                    dim_id='line_state_id', fact_dttm='DateTime', dim_dttm='DateTime', 
                    group_by=['site', 'line'])

                ## aggregating
                mespack_total_rejects_df = mespack_total_rejects_df.assign(DateTimeHour = mespack_total_rejects_df['DateTime'].dt.floor(pack_production_agg_freq))

                groupby_cols = ['site', 'line', 'DateTimeHour', 'tag', 
                                'Machine_Speed', 'Agile_Flag', 'Project_Flag', 'line_state_id']
                agg_dict = {'qty':'sum', 'DateTime':'min'}

                mespack_total_rejects_agg_df = mespack_total_rejects_df.groupby(groupby_cols).agg(agg_dict).reset_index()

                ## transposing
                mespack_total_rejects_agg_df = transpose_data_frame(data_frame=mespack_total_rejects_agg_df,
                                                rows=['site', 'line', 'DateTimeHour', 'Machine_Speed', 'Agile_Flag', 
                                                      'Project_Flag', 'line_state_id', 'DateTime'],
                                                transpose_on='tag')

                # TODO: remove once the whole platform will stop using Changeover_Reject_Total tag
                mespack_total_rejects_agg_df = mespack_total_rejects_agg_df.assign(Changeover_Reject_Total = 0)

                agg_dict = {'Changeover_Reject_Total':'sum', 'Good_Bag_Total':'sum', 
                            'Reject_Bag_Total':'sum', 'DateTime':'min'}
                mespack_total_rejects_agg_df = (mespack_total_rejects_agg_df
                                                .groupby(['site', 'line', 'DateTimeHour', 
                                                          'Machine_Speed', 'Agile_Flag', 'Project_Flag',
                                                          'line_state_id'])
                                                .agg(agg_dict))
                mespack_total_rejects_agg_df = mespack_total_rejects_agg_df.reset_index()

                ## adding line_id and site_id
                rename_dict = {'DateTimeHour':'datetime', 'Machine_Speed':'machine_speed', 
                               'Agile_Flag':'agile_flag', 'Project_Flag':'project_flag', 'DateTime':'start_time'}
                mespack_total_rejects_agg_df = (mespack_total_rejects_agg_df
                                    .merge(lines_dim_df[['line_id', 'line']], on = 'line', how='inner')
                                    .merge(sites_dim_df[['site_id', 'site']], on = 'site', how='inner')
                                    .drop(columns=['site', 'line'])
                                    .rename(columns=rename_dict)
                                )
                
                truncate_date = start_time + timedelta(days=site_history_buffer_days)
                site_sud_id = sites_dim_df.loc[sites_dim_df['site'] == site_name, 'site_id'].iloc[0]
                mespack_total_rejects_agg_df = mespack_total_rejects_agg_df.loc[mespack_total_rejects_agg_df['datetime'] >= truncate_date]

                truncate_query = queries.TRUNCATE_REJECTS_TABLE.format(pack_production_table_nm, site_sud_id, truncate_date)
                execute_db_query(conn_string=datalab_db_conn, query=truncate_query, access_token=datalab_db_access_token)
                insert_df_to_mssql_db(df=mespack_total_rejects_agg_df,
                                    db_conn=datalab_db_conn,
                                    db_table=pack_production_table_nm,
                                    access_token=datalab_db_access_token)

                # updating etl status table 
                extract_overview_df = (mespack_total_rejects_agg_df.groupby(['site_id', 'line_id'])
                                        .agg({'datetime':'max', 'line_state_id':'count'})
                                        .reset_index()
                                        .rename(columns={'line_state_id':'n_records', 'datetime':'start_time'}))
                new_last_available_dttm = extract_overview_df['start_time'].min()

                job_summary_record = get_run_summary_entry(job_id, job_nm, pack_production_table_nm,
                                                            job_step_start_dttm, datetime.now(),
                                                            extract_overview_df['n_records'].sum(),
                                                            np.nan, 'start_time', 
                                                            new_last_available_dttm, np.nan)
                insert_df_to_mssql_db(df=job_summary_record,
                          db_conn=datalab_db_conn,
                          db_table=job_summary_table_nm,
                          access_token=datalab_db_access_token)

            except Exception as e:
                extract_overview_df = pd.DataFrame()

                job_summary_record = get_run_summary_entry(job_id, job_nm, pack_production_table_nm,
                                                            job_step_start_dttm, datetime.now(),
                                                            0, np.nan, 'start_time', 
                                                            np.nan, np.nan, status='failed',
                                                            error_message=str(e))

                insert_df_to_mssql_db(df=job_summary_record,
                          db_conn=datalab_db_conn,
                          db_table=job_summary_table_nm,
                          access_token=datalab_db_access_token)

        elif site_historian_source == 'wonderware':
            raise ValueError('Not implemented')

        else:
            err_msg = 'Incorrect historian system name. Should be one of ("proficy", "wonderware"). Adjust config file.'
            raise ValueError(err_msg)

    else:
        pass


refresh_pbi_dataset(api_url=pbi_refresh_url, app_name=pbi_refresh_app)
