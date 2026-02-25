cd /D F:\Ecosystem Non-OneDrive\Development Area\MVP046 SUD Bag Tool\Scripts
call activate base
python bag_tool_etl_ami.py .\etl_config.toml
call conda deactivate