#!/usr/bin/env python
# coding: utf-8

# # Installations

# In[2]:


# get_ipython().system('pwd')

# # In[3]:


# get_ipython().run_cell_magic('writefile', 'requirements.txt', 'sqlalchemy==1.4.47\nsnowflake-sqlalchemy\nlangchain==0.0.202\nsqlalchemy-aurora-data-api\nPyAthena[SQLAlchemy]==2.25.2\nredshift-connector==2.0.910\nsqlalchemy-redshift==0.8.14\n')


# # In[4]:


# get_ipython().system('pip install -r requirements.txt')


# # Imports

# In[5]:

import os
os.system("export AWS_DEFAULT_REGION=us-east-1")
import langchain 
langchain.__version__


# In[6]:


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool
from pydantic import BaseModel
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, AgentExecutor, ConversationalAgent, AgentOutputParser, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
import ast
import re
import argparse
# from tabulate import tabulate


# In[7]:


import json
import boto3

import sqlalchemy
from sqlalchemy import create_engine
# from snowflake.sqlalchemy import URL

from langchain.docstore.document import Document
from langchain import PromptTemplate, SagemakerEndpoint, SQLDatabase, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
# from langchain.chains import SQLDatabaseSequentialChain

from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

from langchain.chains.api import open_meteo_docs

from langchain.llms.bedrock import Bedrock

from typing import Dict, List, Union

import string

from langchain.llms.sagemaker_endpoint import LLMContentHandler

from langchain.memory import ConversationBufferMemory

region = 'us-east-1'


# ## Defining Athena connections

# In[8]:





# In[9]:


# glue_databucket_name, gdc


# In[10]:


def parse_catalog():
    #Connect to Glue catalog
    #get metadata of redshift serverless tables
    columns_str='Source | Database | Table | Column_Name\n'
    
    #define glue cient
    glue_client = boto3.client('glue')
    
    for db in gdc:
        response = glue_client.get_tables(DatabaseName=db)
        for tables in response['TableList']:
            #classification in the response for s3 and other databases is different. Set classification based on the response location
            if tables['StorageDescriptor']['Location'].startswith('s3'):  
                classification='s3' 
            else:  
                classification = tables['Parameters']['classification']
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname, tblname, colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str = columns_str + f'{classification} | {dbname} | {tblname} | {colname}\n'                     
    # #API
    # ## Append the metadata of the API to the unified glue data catalog
    columns_str=columns_str+ 'api | meteo | weather | weather\n'
    return columns_str






class ContentHandlerFlanT5XL(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt, model_kwargs) :
        input_json = {"text_inputs": prompt, **model_kwargs}
        encoded_json = json.dumps(input_json).encode("utf-8")
        return encoded_json
    
    def transform_output(self, output):
        response = json.loads(output.read().decode("utf-8")).get('generated_texts')
        # print("response" , response)
        return "".join(response)
    
    
    
class ContentHandlerFalcon(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt, model_kwargs) :
        input_json = {"inputs": prompt, "parameters": model_kwargs}
        encoded_json = json.dumps(input_json).encode("utf-8")
        return encoded_json
    
    def transform_output(self, output):
        response = json.loads(output.read().decode("utf-8"))[0]['generated_text']
        # print("response" , response)
        return response



glue_crawler_name = 'life_insurance'
glue_database_name = 'life_insurance'
glue_databucket_name = 'sagemaker-studio-696784033931-jq9nmfb0t5c'


## athena variables
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_database_name #from cfn params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)


#--------------------------------------------------------------

#AURORA MYSQL
##connect to aurora mysql
##aurora mysql cluster details/variables
cluster_arn = "arn:aws:rds:us-east-1:696784033931:cluster:database-1"
secret_arn = "arn:aws:secretsmanager:us-east-1:696784033931:secret:rds-db-credentials/cluster-BUBG6BLYPA57SUIJFTAFJDIHJU/admin/1688467914091-9poxQk"
rdsdb="titanic" # genai_test

rdsdb_tbl = ["titanic_dataset"]
##  Create the aurora connection string
connection_string = f"mysql+auroradataapi://:@/{rdsdb}"
##  Create the aurora  SQLAlchemy engine
engine_rds = create_engine(connection_string, echo=False,connect_args=dict(aurora_cluster_arn=cluster_arn, secret_arn=secret_arn))
dbrds = SQLDatabase(engine_rds, include_tables=rdsdb_tbl)


gdc = [schemaathena] 

glue_catalog = parse_catalog()

#display a few lines from the catalog
#print('\n'.join(glue_catalog.splitlines()))


columns = ["PassengerId"	,"Survived",	"Pclass",	"Name",	"Sex",	"Age",	"SibSp",	"Parch",	"Ticket",	"Fare"	,"Cabin",	"Embarked","premium_amount" ]
glue_catalog_rds = ""
for col in columns:
    glue_catalog_rds = glue_catalog_rds + f"rdsmysql | titanic | titanic_dataset | {col}\n"

columns = ["Name",	"Sex",	"Age", "premium_amount" ]
for col in columns:
    glue_catalog_rds = glue_catalog_rds + f"rdsmysql | titanic | customers | {col}\n"

glue_catalog = glue_catalog + glue_catalog_rds


    
boto3_bedrock  = boto3.client(
    service_name="bedrock",
    region_name="us-east-1"
)
bedrock_llm = Bedrock(
    model_id="anthropic.claude-v1", 
    client=boto3_bedrock, 
    model_kwargs={
        'temperature': 0, 
        'max_tokens_to_sample': 5000,
        # 'top_k': 10,
        # 'top_p': 1,
        # "stop_sequences": [
        #     "Observation:"#, "Human:"
        # ]
    }
)



# # Functions

# In[14]:


def extarct_column_information_for_s3_athena(table_name, database_name, db_connector):
    response = db_connector.run(f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{database_name}';")
    response = ast.literal_eval(response)
    column_data = 'COLUMN_NAME | DATA_TYPE | IS_NULLABLE\n'
    for _r in response:
        column_data += f'{_r[0]} | {_r[1]} | {_r[2]}\n'
    return column_data

def extarct_column_information_for_rds(table_name, database_name, db_connector):
    return ''

def identify_channel(query):
    #Prompt 1 'Infer Channel'
    ##set prompt template. It instructs the llm on how to evaluate and respond to the llm. It is referred to as dynamic since glue data catalog is first getting generated and appended to the prompt.
    prompt_template = """
Question: {query}
Reference Table: 
\n
{catalog}
\n
Reference Table has following 4 columns, 
1. Source : information on where the our data is stored.
2. Database : name of the database which is related to the question
3. Table : name of the table which answers the question.
4. Column_Name : name of the column which answers the question.

Human: Use the Reference Table and find the 'source_name' (in column Source), 'database_name' (in column Database), 'table_name' (in column Table) and 'column_name' (in column Column_Name) which is most related to the Question.
Answer returned from AI should be syntactically correct json with keys source_name, database_name, table_name and column_name with their respective values.
Return just the answer and nothing else. JSON should be complete, without any syntax error.
AI: 
"""
#     Please return a syntactically correct json string with keys as source, database, table and column and they should have the value extracted from the Table specifically source_name, database_name and table_name, column_name.
# 
    ##define prompt 1
    prompt = PromptTemplate(template=prompt_template, input_variables=["catalog", "query"])

    # define llm chain
    llm_chain = LLMChain(prompt=prompt, llm=bedrock_llm)  # only bedrock works best here
    #run the query and save to generated texts
    generated_texts = llm_chain.run({
        'catalog': glue_catalog,
        'query': query,
    })
    # print(generated_texts)
    #set the best channel from where the query can be answered
    if 'snowflake' in generated_texts: 
        channel='snowflake'
        db=dbsnowflake 
        # print("SET database to snowflake")  
    elif 'redshift'  in generated_texts: 
        channel='redshift'
        db=dbredshift
        # print("SET database to redshift")
    elif 's3' in generated_texts: 
        channel='s3'
        db=dbathena
        # print("SET database to athena")
    elif 'rdsmysql' in generated_texts: 
        channel='rdsmysql'
        db=dbrds
        # print("SET database to rds")    
    elif 'api' in generated_texts: 
        channel='api'
        # print("SET database to weather api")        
    else: 
        raise Exception("User question cannot be answered by any of the channels mentioned in the catalog")
    
    
    # cleaning any additional generations
    _, _, tail = generated_texts.partition('{')
    head, _, _ = tail.partition('}')
    json_string = '{' + head + '}'
    json_data = json.loads(json_string)
    #print(f"Json from Database LLM call: {json_string}")
    
    database_name = json_data['database_name']
    table_name = json_data['table_name']
    column_data = ''
    if channel == 's3':
        column_data = extarct_column_information_for_s3_athena(table_name, database_name, db)
    elif channel == 'rdsmysql':
        column_data = extarct_column_information_for_rds(table_name, database_name, db)
    else:
        column_data = ''
        
    #print(100*"-")
    return channel, db, database_name, table_name, column_data

def valid_sql_keywords(dialect):
    if dialect == 'awsathena':
        return """ALTER, AND, AS, BETWEEN, BY, CASE, CAST,
CONSTRAINT, CREATE, CROSS, CUBE, CURRENT_PATH, CURRENT_USER, DEALLOCATE, 	
DELETE, DESCRIBE, DISTINCT, DROP, ELSE, END, ESCAPE, EXCEPT, 	
EXECUTE, EXISTS, EXTRACT, FALSE, FIRST, FOR, FROM, FULL, GROUP, 	
GROUPING, HAVING, IN, INNER, INSERT, INTERSECT, INTO, 	
IS, JOIN, LAST, LEFT, LIKE, LOCALTIME, LOCALTIMESTAMP, NATURAL, 
NORMALIZE, NOT, NULL, OF, ON, OR, ORDER, OUTER, PREPARE, 
RECURSIVE, RIGHT, ROLLUP, SELECT, SKIP, TABLE, THEN, TRUE, 	
UNESCAPE, UNION, UNNEST, USING, VALUES, WHEN, WHERE, WITH""" # CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP,
    if dialect == 'mysql': # refers to rds mysql
        return """ALTER, AND, AS, BETWEEN, BY, CASE, CAST,
CONSTRAINT, CREATE, CROSS, CUBE, CURRENT_PATH, CURRENT_USER, DEALLOCATE, 	
DELETE, DESCRIBE, DISTINCT, DROP, ELSE, END, ESCAPE, EXCEPT, 	
EXECUTE, EXISTS, EXTRACT, FALSE, FIRST, FOR, FROM, FULL, GROUP, 	
GROUPING, HAVING, IN, INNER, INSERT, INTERSECT, INTO, 	
IS, JOIN, LAST, LEFT, LIKE, LOCALTIME, LOCALTIMESTAMP, NATURAL, 
NORMALIZE, NOT, NULL, OF, ON, OR, ORDER, OUTER, PREPARE, 
RECURSIVE, RIGHT, ROLLUP, SELECT, SKIP, TABLE, THEN, TRUE, 	
UNESCAPE, UNION, UNNEST, USING, VALUES, WHEN, WHERE, WITH""" # CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP,  # to implement
    else:
        return ''
    
def generate_sql_query(query, channel, db_connector, database, table_name, column_data):
    _DEFAULT_TEMPLATE = """
Question: {question}

To answer the above question, create a syntactically correct {dialect} Query using Reserved Keywords given below, for the Database: {database_name} and Table Name: {table_name}.
Created query should be using the columns related to the question only. The following table describes valid columns names in the table of the database, data type for each column name and whether that column can have null values is given below:
{column_data}

Reserved Keywords that can be used to build SQL SELECT statements and queries for {dialect} are following: 
{sql_reserved_keywords}
Human: Return the generated {dialect} query using the reserved keywords which is syntactically correct. Please write the query in <sql></sql> tags.
AI:
"""
    
    prompt = PromptTemplate(
        input_variables=["question", "dialect", "database_name", "table_name", "column_data", "sql_reserved_keywords"], template=_DEFAULT_TEMPLATE
    )
    
    llm_chain = LLMChain(
        prompt=prompt, 
        llm=bedrock_llm
    )

    generated_texts = llm_chain.run({
        'question': query,
        'dialect': db_connector.dialect,
        'database_name': database,
        'table_name': table_name,
        'column_data': column_data,
        'sql_reserved_keywords': valid_sql_keywords(db_connector.dialect)
    })
    #print(f"SQL LLM Call: {generated_texts}")
    #print(100*"-")
    
    sql_query = generated_texts.split('l>')[1].split('</s')[0]
    return sql_query

def validate_query(sql_query, question, database_name, table_name, column_data):
    _DEFAULT_TEMPLATE = """
Question: {question}

Table Name: {table_name}

Database Name: {database_name}

The following table describes valid columns names in the table of the database, data type for each column name and whether that column can have null values:
{column_data}

SQL Query: {sql_query}

Human: Verify the SQL Query to check if it answers the Question using the related columns only and no additional conditions are added. Please rewrite the query in <sql></sql> tags and return the correct query.
AI:
"""
    
    prompt = PromptTemplate(
        input_variables=["question", "table_name", "database_name", "column_data", "sql_query"], template=_DEFAULT_TEMPLATE
    )
    
    llm_chain = LLMChain(
        prompt=prompt, 
        llm=bedrock_llm
    )

    generated_texts = llm_chain.run({
        'question': question,
        'table_name': table_name,
        'database_name': database_name,
        'column_data': column_data,
        'sql_query': sql_query,
    })
    #print(f"Verification SQL LLM Call: {generated_texts}")
    #print(100*"-")
    
    sql_query = generated_texts.split('l>')[1].split('</s')[0]
    return sql_query
    
    
def run_query(query):
    channel, db_connector, database, table_name, column_data = identify_channel(query)
    
    sql_query = generate_sql_query(query, channel, db_connector, database, table_name, column_data)
    
    
#     sql_query_validator_response = sqlvalidator.parse(sql_query)

#     if not sql_query_validator_response.is_valid():
#         print(f'Error with the SQL Statement:\n {sql_query_validator_response.errors}')
#         print("Re trying to generate the Query...")
#         query += f'\n Human: Strictly Do not generate invalid query like {sql_query}'
#         sql_query = generate_sql_query(query, channel, db_connector, database, table_name, column_data)
        
    sql_query = validate_query(sql_query, query, database, table_name, column_data)
    sql_results = db_connector.run(sql_query)
    
    return sql_results, sql_query



def main(query):



    

    #Response from Langchain
    response, sql_query =  run_query(query)
    # print("----------------------------------------------------------------------")
    # print(f"Query: {query}")    
    print("----------------------------------------------------------------------")
    print(f"Response: {response}")
    # print(tabulate(response))


if __name__=='__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--query")

    args = parser.parse_args()
    main(args.query)
