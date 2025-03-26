# Databricks notebook source
# MAGIC %md 
# MAGIC # init notebook setting up the backend. 
# MAGIC
# MAGIC Do not edit the notebook, it contains import and helpers for the demo
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=00-init&demo_name=chatbot-rag-llm&event=VIEW">

# COMMAND ----------

# %run ../config_ooliblikas

# COMMAND ----------


# dbutils.widgets.text("reset_all_data", "false", "Reset Data")
# reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
dbutils.widgets.text("recreate_source_table", "false", "Recreate Source Table databricks_scraped_taastuvenergia")
recreate_source_table = dbutils.widgets.get("recreate_source_table") == "true"

# COMMAND ----------

from IPython.core.magic import register_cell_magic

# When running in a job, writting to the local file fails. This simply skip it as we ship the file in the repo so there is no need to write it if it's running from the repo
# Note: we don't ship the chain file when we install it with dbdemos.instal(...).
@register_cell_magic
def writefile(line, cell):
    filename = line.strip()
    try:
      folder_path = os.path.dirname(filename)
      if len(folder_path) > 0:
        os.makedirs(folder_path, exist_ok=True)
      with open(filename, 'w') as f:
          f.write(cell)
          print('file overwritten')
    except Exception as e:
      print(f"WARN: could not write the file {filename}. If it's running as a job it's to be expected, otherwise something is off - please print the message for more details: {e}")

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
import yaml
from typing import Iterator
from mlflow import MlflowClient
mlflow.set_registry_uri('databricks-uc')

# Workaround for a bug fix that is in progress
mlflow.spark.autolog(disable=True)

import warnings
warnings.filterwarnings("ignore")
# Disable MLflow warnings
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)

# COMMAND ----------

#dbdemos__delete_this_cell
#force the experiment to the field demos one. Required to launch as a batch
def init_experiment_for_batch(demo_name, experiment_name):
  #You can programatically get a PAT token with the following
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  xp_root_path = f"/Shared/dbdemos/experiments/{demo_name}"
  try:
    r = w.workspace.mkdirs(path=xp_root_path)
  except Exception as e:
    print(f"ERROR: couldn't create a folder for the experiment under {xp_root_path} - please create the folder manually or  skip this init (used for job only: {e})")
    raise e
  xp = f"{xp_root_path}/{experiment_name}"
  print(f"Using common experiment under {xp}")
  mlflow.set_experiment(xp)

# COMMAND ----------

# if reset_all_data:
#   print(f'clearing up db {dbName}')
#   spark.sql(f"DROP DATABASE IF EXISTS `{dbName}` CASCADE")

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

assert catalog not in ['hive_metastore', 'spark_catalog'], "Please use a UC schema"
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
      if catalog == 'dbdemos':
        spark.sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
  use_and_create_db(catalog, dbName)


print(f"using catalog.database `{catalog}`.`{dbName}`")
spark.sql(f"""USE `{catalog}`.`{dbName}`""")    

# COMMAND ----------

# DBTITLE 1,Optional: Allowing Model Serving IPs
#If your workspace has ip access list, you need to allow your model serving endpoint to hit your AI gateway. Based on your region, IPs might change. Please reach out your Databrics Account team for more details.

#def allow_serverless_ip():
#  from databricks.sdk import WorkspaceClient
#  from databricks.sdk.service import settings
#
#  w = WorkspaceClient()
#
#  # cleanup
#  w.ip_access_lists.delete(ip_access_list_id='xxxx')
#  created = w.ip_access_lists.create(label=f'serverless-model-serving',
#                                    ip_addresses=['xxxx/32'],
#                                    list_type=settings.ListType.ALLOW)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Helpers to get catalog and index status:

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# DBTITLE 1,endpoint
import time

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,index
def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

def wait_for_model_serving_endpoint_to_be_ready(ep_name):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # TODO make the endpoint name as a param
    # Wait for it to be ready
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(ep_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch content of keskkonnaportaal and its subpages
# MAGIC Save the results to unity catalog
# MAGIC
# MAGIC id | url | content | 
# MAGIC -- | -- | --| 
# MAGIC 0 | https://keskkonnaportaal.ee/et/teemad/taastuvenergia/paikeseenergia | Päikeseenergia ... |
# MAGIC 1 | https://keskkonnaportaal.ee/et/teemad/taastuvenergia/bioenergia | Bioenergia ... |
# MAGIC 2 | https://keskkonnaportaal.ee/et/teemad/taastuvenergia/tuuleenergia/infraheli | Infraheli ... |

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://keskkonnaportaal.ee/et/teemad/taastuvenergia"

# Add retries with backoff to avoid 429 errors
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)

# def get_all_links(base_url):
#     """Fetches all links from the given base URL"""
#     response = requests.get(base_url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     links = set()
#     for a_tag in soup.find_all("a", href=True):
#         url = a_tag["href"]
#         if url.startswith("/"):
#             url = "https://keskkonnaportaal.ee" + url  # Convert relative to absolute URL
#         if url.startswith("https://keskkonnaportaal.ee/et/teemad/taastuvenergia"):
#             links.add(url)

#         # not to get too many the links
#         # if len(links) > 4:
#         #     break
#     return list(links)

# # Fetch all links from the website
# all_urls = get_all_links(BASE_URL)
# # print("Number of links:", len(all_urls), all_urls)

# df_urls = spark.createDataFrame(all_urls, StringType()).toDF("url").repartition(10)

# # Pandas UDF to fetch HTML content for a batch of URLs
@pandas_udf("string")
def fetch_html_udf(urls: pd.Series) -> pd.Series:
    adapter = HTTPAdapter(max_retries=retries)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)
    
    def fetch_html(url):
        try:
            response = http.get(url)
            if response.status_code == 200:
                return response.content
        except requests.RequestException:
            return None
        return None
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(fetch_html, urls))
    return pd.Series(results)

# Pandas UDF to extract text from HTML content
@pandas_udf("string")
def extract_text_udf(html_contents: pd.Series) -> pd.Series:
    def extract_text(html_content):
        if html_content:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ").strip()
        return None
    
    return html_contents.apply(extract_text)

# # Apply UDFs to process data
# df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
# final_df = df_with_html.withColumn("content", extract_text_udf("html_content"))

# # Select and filter non-null results
# # final_df = final_df.select("url", "content").filter("content IS NOT NULL")
# final_df = final_df.selectExpr("monotonically_increasing_id() as id", "url", "content").filter("content IS NOT NULL")

# # Save to Unity Catalog
# final_df.write.mode('overwrite').saveAsTable('databricks_scraped_taastuvenergia')
# spark.sql('ALTER TABLE databricks_scraped_taastuvenergia SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

# display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract text from images

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Extract image urls

# COMMAND ----------

def get_all_links_and_images(base_url):
    """Fetches all links and image URLs from the given base URL"""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    links = set()
    image_urls = set()

    for a_tag in soup.find_all("a", href=True):
        url = a_tag["href"]
        if url.startswith("/"):
            url = "https://keskkonnaportaal.ee" + url
        if url.startswith("https://keskkonnaportaal.ee/et/teemad/taastuvenergia"):
            links.add(url)

    # Extract images
    for img_tag in soup.find_all("img", src=True):
        img_url = img_tag["src"]
        if img_url.startswith("/"):
            img_url = "https://keskkonnaportaal.ee" + img_url
        image_urls.add(img_url)

    return list(links), list(image_urls)

if recreate_source_table:
    # Fetch all links and images
    all_urls, all_image_urls = get_all_links_and_images(BASE_URL)

    df_urls = spark.createDataFrame(all_urls, StringType()).toDF("website_url").repartition(10)
    df_images = spark.createDataFrame(all_image_urls, StringType()).toDF("image_url").repartition(10)

    print(f"There are {len(all_urls)} urls and {len(all_image_urls)} image urls")
    display(df_urls.limit(10))
    display(df_images.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Process image urls with OCR
# MAGIC * Download image
# MAGIC * Preprocess using opencv
# MAGIC * Extract text using pytesseract
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet opencv-python==4.10.0.84

# COMMAND ----------

import pytesseract
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError

@pandas_udf("string")
def extract_text_from_images(image_urls: pd.Series) -> pd.Series:
    def ocr_image(image_url):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                try:
                    image = Image.open(BytesIO(response.content))
                except UnidentifiedImageError:
                    print("Cannot identify image file: {image_urk}")
                    return None
                image = np.array(image.convert("L"))  # Convert to grayscale
                
                # Preprocessing with OpenCV
                _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                text = pytesseract.image_to_string(image, lang="eng+est")  # English + Estonian OCR
                return text.strip() if text else None
        except requests.RequestException:
            return None
        return None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(ocr_image, image_urls))
    return pd.Series(results)


if recreate_source_table:
    # apply OCR UDF
    df_with_image_text = df_images.withColumn("image_text", extract_text_from_images("image_url"))

    # keep not null columns
    df_with_image_text = df_with_image_text.filter(df_with_image_text.image_text.isNotNull())

    print(f"Images with extracted text")
    display(df_with_image_text)


# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Combine text from web page and image

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id


if recreate_source_table:
    print(f"Fetching html content for all URLs")
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("website_url"))

    print(f"Extracting text from HTML content")
    final_df = df_with_html.withColumn("content", extract_text_udf("html_content"))

    print(f"Creating final dataset by selecting necessary columns and non null results")
    final_df = final_df.selectExpr("website_url as url", "content").filter("content IS NOT NULL")

    final_text_df = final_df.unionByName(
        df_with_image_text.withColumnRenamed("image_url", "url").withColumnRenamed("image_text", "content"),
        allowMissingColumns=True
    ).withColumn("id", monotonically_increasing_id())

    print(f"Saving image text + website text data to unity catalog")
    final_text_df.write.mode('overwrite').saveAsTable('databricks_scraped_taastuvenergia_with_images')
    spark.sql('ALTER TABLE databricks_scraped_taastuvenergia_with_images SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

    display(final_text_df)


# COMMAND ----------

def display_gradio_app(space_name = "databricks-demos-chatbot"):
    displayHTML(f'''<div style="margin: auto; width: 1000px"><iframe src="https://{space_name}.hf.space" frameborder="0" width="1000" height="950" style="margin: auto"></iframe></div>''')

# COMMAND ----------

#Display a better quota message 
def display_quota_error(e, ep_name):
  if "QUOTA_EXCEEDED" in str(e): 
    displayHTML(f'<div style="background-color: #ffd5b8; border-radius: 15px; padding: 20px;"><h1>Error: Vector search Quota exceeded in endpoint {ep_name}</h1><p>Please select another endpoint in the ../config file (VECTOR_SEARCH_ENDPOINT_NAME="<your-endpoint-name>"), or <a href="/compute/vector-search" target="_blank">open the vector search compute page</a> to cleanup resources.</p></div>')

# COMMAND ----------

# DBTITLE 1,Cleanup utility to remove demo assets
def cleanup_demo(catalog, db, serving_endpoint_name, vs_index_fullname):
  vsc = VectorSearchClient()
  try:
    vsc.delete_index(endpoint_name = VECTOR_SEARCH_ENDPOINT_NAME, index_name=vs_index_fullname)
  except Exception as e:
    print(f"can't delete index {VECTOR_SEARCH_ENDPOINT_NAME} {vs_index_fullname} - might not be existing: {e}")
  try:
    WorkspaceClient().serving_endpoints.delete(serving_endpoint_name)
  except Exception as e:
    print(f"can't delete serving endpoint {serving_endpoint_name} - might not be existing: {e}")
  spark.sql(f'DROP SCHEMA `{catalog}`.`{db}` CASCADE')

# COMMAND ----------

def pprint(obj):
  import pprint
  pprint.pprint(obj, compact=True, indent=1, width=100)
