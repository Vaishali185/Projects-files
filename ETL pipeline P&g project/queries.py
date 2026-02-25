# This file contains the SQL repository for the Packing Overspeed tool

# HISTORIAN

HISTN_SELECT_DELTA = """
	SELECT TagName
          ,DateTime
          ,Value
    FROM History
    WHERE TagName IN {}
      AND DateTime BETWEEN '{}' AND '{}'
      AND wwVersion = 'Latest'
      AND wwRetrievalMode = 'Delta'
"""

HISTN_SELECT_DELTA_VVALUE = """
	SELECT TagName
          ,DateTime
          ,Value
          ,vValue
    FROM History
    WHERE TagName IN {}
      AND DateTime BETWEEN '{}' AND '{}'
      AND wwVersion = 'Latest'
      AND wwRetrievalMode = 'Delta'
"""

HISTN_SELECT_CYCLYC = """
	SELECT TagName
          ,DateTime
          ,Value
    FROM History
    WHERE TagName IN {}
      AND DateTime BETWEEN '{}' AND '{}'
      AND wwResolution = {}
      AND wwVersion = 'Latest'
      AND wwRetrievalMode = 'Cyclic'
"""

# DATALAB

EXTRACT_LINE_STATE = """
    SELECT ls.line_state_id
        ,ls.start_time
        ,ls.end_time
        ,l.line
        ,s.site
    FROM SUD_LINE_STATE as ls
    INNER JOIN SUD_LINES as l 
        ON ls.line_id = l.line_id 
    INNER JOIN SUD_SITES as s
        ON ls.site_id = s.site_id
    WHERE ls.start_time >= '{}'
    AND l.line in {}
"""

CREATE_BAG_REJECTS_TMP = """

drop table if exists {};

create table {}(
	[line_state_id] [nvarchar](40) NOT NULL,
	[site_id] [int] NOT NULL,
	[line_id] [int] NOT NULL,
	[datetime] [datetime] NOT NULL,
	[reject_type] [nvarchar](40) NOT NULL,
	[machine_speed] [int] NOT NULL,
    [agile_flag] [int] NOT NULL,
    [project_flag] [int] NOT NULL,
	[start_time] [datetime] NULL,
	[rejects_qty] [int] NULL
);
"""

CREATE_BAG_TOTALS_TMP = """

drop table if exists {};

create table {}(
	[line_state_id] [nvarchar](40) NOT NULL,
	[site_id] [int] NOT NULL,
	[line_id] [int] NOT NULL,
	[datetime] [datetime] NOT NULL,
	[machine_speed] [int] NOT NULL,
    [agile_flag] [int] NOT NULL,
    [project_flag] [int] NOT NULL,
	[start_time] [datetime] NULL,
	[Good_Bag_Total] [int] NULL,
	[Reject_Bag_Total] [int] NULL,
	[Changeover_Reject_Total] [int] NULL,
	[Camera_DS_Reject_Total] [int] NULL,
	[Camera_US_Reject_Total] [int] NULL
);
"""

TRUNCATE_TAG_LIST_TABLE = """ 
    delete from {}
    where Site = '{}' and ToolName='{}'
"""

TRUNCATE_REJECTS_TABLE = """ 
    delete from {}
    where site_id = {} and datetime >= '{}'
"""


# COMMON
DB_DELETE_ROWS_WHERE_ONE_COLUMN = """
    DELETE FROM {} WHERE {} {} {};
"""

DB_DELETE_FROM_TABLE = """
    DELETE FROM {};
"""

DB_DELETE_FROM_WHERE = """
    DELETE FROM {}
    WHERE {} {} {};
"""

DB_TRUNCATE_BY_TIME = """
    DELETE FROM {}
    WHERE {} {} '{}';
"""

DB_DROP_TABLE = """
    DROP TABLE {};
"""

DB_GET_LAST_DTTM = """
    SELECT MAX({}) FROM {};
"""

DB_GET_LAST_DTTM_SITE = """
    SELECT MAX({}) FROM {} WHERE SITE = '{}';
"""

DB_UPSERT_MERGE = """
    MERGE {} AS target
    USING {} AS source
    ON ({})
    WHEN MATCHED THEN
    UPDATE SET {}
    WHEN NOT MATCHED BY TARGET
    THEN INSERT ({}) VALUES ({})
    ;
    """
    
DB_UPDATE_VALUE_WHERE = """
    UPDATE {} 
    SET {} = {}
    WHERE {} = {}
    ;
 """