import duckdb
import pandas as pd

# Load CSVs into DuckDB
# conn = duckdb.connect("data/sam.duckdb")
# df = pd.read_csv("data/sam_contract_ops_071225.csv", encoding="ISO-8859-1")
# conn.execute("CREATE TABLE sam_contract_ops AS SELECT * FROM df")

conn = duckdb.connect("data/sam.duckdb")
df_archived = pd.read_csv(
    "data/sam_contracts_archived_2024.csv", encoding="ISO-8859-1", dtype="str")
conn.execute("create or replace table sam_contracts_archived AS SELECT * FROM df_archived")
conn.execute("""
                create or replace view contracts AS 
                select "Notice ID" as id, title, Description as description, "Contract Opportunity Type" as contract_type,NAICS as classification,PSC as prod_serv_code, "Contract Award Date" as contract_award_dt from sam_contract_ops
                where Description is not null;
            """)
