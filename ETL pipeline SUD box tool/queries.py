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

CREATE_BOX_REJECTS = """
drop table if exists {};

create table {}(
    [datetime] [datetime] NOT NULL,
    [agile_flag] [int] NOT NULL,
    [recipe_id] [int] NOT NULL,
    [line_state_id] [nvarchar](40) NOT NULL,
    [start_time] [datetime] NULL,
    [rejects_qty] [int] NULL,
    [line_id] [int] NOT NULL,
    [site_id] [int] NOT NULL,
    [reject_cause_id] [int] NOT NULL
);
"""


CREATE_BOX_REJECTS_CAUSE_DIM = """
drop table if exists {};

create table {}(
    [reject_cause_id] [int] IDENTITY(1,1) PRIMARY KEY,
    [tag_nm] [nvarchar](150) NOT NULL,
    [cause] [nvarchar](100) NULL,
    [reason] [nvarchar](40) NULL,
    [station] [nvarchar](100) NOT NULL,
    [type] [nvarchar](40) NOT NULL,
    [machine] [nvarchar](40) NOT NULL
    
);
"""


CREATE_BOX_LINE_RECIPE_DIM = """
drop table if exists {};

create table {}(
    [line_recipe_id] [int]  PRIMARY KEY,
    [product_segment] [nvarchar](100) NOT NULL,
    [recipe_size_nm] [nvarchar](40) NOT NULL,
    [recipe_cd] [nvarchar](40) NOT NULL,
    [n_boxes_per_case] [int] NOT NULL,
    
    
    
);
"""

CREATE_BOX_TOTALS = """
drop table if exists {};

create table {}(
       [site_id] [int] NOT NULL,
       [line_id] [int] NOT NULL, 
       [datetime] [datetime] NOT NULL, 
       [recipe_id] [int] NOT NULL,
       [agile_flag] [int] NOT NULL,
       [line_state_id] [nvarchar](40) NOT NULL,
       [Base_Machine_ExtractedCartons_0_Counter_Actual_n] [int] NULL,
       [Base_Machine_ProducedBases_0_Counter_Actual_n] [int] NULL,
       [Cover_General_ExtractedCartons_Total_Counter_Actual_n] [int] NULL,
       [Cover_General_ProducedCovers_Total_Counter_Actual_n] [int] NULL,
       [Base_Machine_RejectedBases_Station1_Counter_Actual_n] [int] NULL,
       [Base_Machine_RejectedBases_Station2_Counter_Actual_n] [int] NULL,
       [Base_Machine_RejectedBases_Station3_Counter_Actual_n] [int] NULL,
       [Base_Machine_RejectedBases_Total_NOT_Extracted_Betw_Holders_n] [int] NULL,
       [Cover_General_RejectedCovers_0_Counter_Actual_n] [int] NULL,
       [UPack_For_Proficy_Produced_Cases] [int] NULL,
       [start_time] [datetime] NULL

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