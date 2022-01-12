from datetime import datetime
import json
import os
import time
from typing import Any, Callable, Dict, NamedTuple, Optional
from IPython.display import clear_output

from google import auth
from google.api_core import exceptions as google_exceptions
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google_cloud_pipeline_components.experimental import forecasting as gcc_aip_forecasting
import google.cloud.aiplatform
from google.cloud import bigquery
from google.cloud import storage

from google.colab import auth as colab_auth
from google.colab import drive

import kfp
import kfp.v2.dsl
from kfp.v2.google import client as pipelines_client

from matplotlib import dates as mdates
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns

from IPython.display import Image
from IPython.core.display import HTML 
import args_parse 

FLAGS = args_parse.parse_common_options(
  PROJECT_ID=None,
  LOCATION=None,
  VERSION=None,
  OVERRIDE='True',
  VERSION=None,
  PIPELINE_TAG=None,
  PACKAGE_PATH=None,
  OVERWRITE=None,
  PIPELINE_LOCATION = 'us-central1',
  BATCH_JOB_LOCATION = 'us-central1',
  TRAIN_PIPELINE_NAME = 'lowes-2022-v1-train-no-combo-20220111165836', 
  HISTORY_WINDOW_n = 13, 
  FORECAST_HORIZON = 13,  
  EVAL_DESTINATION_DATASET = 'lowes_forecast_22_v1',
  PIPELINES_FILEPATH = 'gs://lowes-vertex-forecast-poc/pipelines/pipelines.json',
  FORECAST_PRODUCTS_TABLE = 'jtotten-project.jtott_lowes.products',
  FORECAST_ACTIVITIES_TABLE = 'jtotten-project.jtott_lowes.jt_pm_holdout_13wks_v3',
  FORECAST_PLAN_TABLE ='jtotten-project.jtott_lowes.jt_plan_pm_holdout_13wks_v3',
  GS_PIPELINE_ROOT_PATH = 'gs://jtott/pipeline_root/lowes-vertex-forecast-poc',
  FORECAST_ACTIVITIES_EXPECTED_HISTORICAL_LAST_DATE = '2020-10-02',
  TIME_GRANULARITY_UNIT = 'WEEK',
  TIME_GRANULARITY_QUANTITY = 1,
  JSON_PACKAGE_PATH='custom_container_pipeline_spec.json',
  
)

PIPELINES = {}

if os.path.isfile(FLAGS.PIPELINES_FILEPATH):
  with open(FLAGS.PIPELINES_FILEPATH) as f:
    PIPELINES = json.load(f)
else:
  PIPELINES = {}
  
 
##### Create forecast input table
@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.7.0'],
)
def create_forecast_input_table_specs(
  project: str,
  pipeline_location: str,
  train_pipeline_name: str,
  forecast_products_table_uri: str,
  forecast_activities_table_uri: str,
  forecast_plan_table_uri: str,
) -> NamedTuple('Outputs', [('forecast_input_table_specs', str)]):
  import json
  import os
  from google.cloud.aiplatform import pipeline_jobs

  train_pipeline_job = pipeline_jobs.PipelineJob.get(
    resource_name=os.path.basename(train_pipeline_name),
    project=project,
    location=pipeline_location,
  ).to_dict()

  input_table_specs = json.loads(
    next(
      task_details
      for task_details in train_pipeline_job['jobDetail']['taskDetails']
      if task_details['taskName'] == 'create-input-table-specs'
    )
    ['execution']
    ['metadata']
    ['output:input_table_specs']
  )

  forecast_input_table_specs = [
    {
      'bigquery_uri': forecast_plan_table_uri,
      'table_type': 'FORECASTING_PLAN',
    },
  ]
  for table_spec in input_table_specs:
    if table_spec['table_type'] == 'FORECASTING_PRIMARY':
      table_uri = forecast_activities_table_uri
    else:
      table_uri = forecast_products_table_uri
    table_spec['bigquery_uri'] = table_uri
    forecast_input_table_specs.append(table_spec)

  print(forecast_input_table_specs)
  
  return (json.dumps(forecast_input_table_specs),)  # forecast_input_table_specs

##### Get predict table
@kfp.v2.dsl.component(base_image='python:3.9')
def get_predict_table_path(
  predict_processed_table: str
) -> NamedTuple('Outputs', [('preprocess_bq_uri', str)]):
  import json

  preprocess_bq_uri = (
    json.loads(predict_processed_table)
    ['processed_bigquery_table_uri']
  )
  return (preprocess_bq_uri,)

##### Retrieve models metadata
@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.7.0'],
)
def get_models_metadata(
  project: str,
  location: str,
  train_pipeline_name: str,
) -> NamedTuple('Outputs', [('models_metadata', str)]):
  import argparse
  import json
  import os

  from google.cloud.aiplatform import pipeline_jobs

  args_parser = argparse.ArgumentParser()
  args_parser.add_argument(
    '--method.model_display_name',
    dest='model_display_name',
    required=True,
  )

  train_pipeline_job = pipeline_jobs.PipelineJob.get(
    resource_name=os.path.basename(train_pipeline_name),
    project=project,
    location=location,
  ).to_dict()

  train_job_to_model_display_name = {}
  executors = (
    train_pipeline_job['pipelineSpec']['deploymentConfig']['executors']
  )
  for executor_key, executor_spec in executors.items():
    if executor_key.startswith('exec-automlforecastingtrainingjob-run'):
      args = executor_spec['container']['args']
      parsed_args, _ = args_parser.parse_known_args(args)
      task_name = executor_key[5:]
      train_job_to_model_display_name[task_name] = (
        parsed_args.model_display_name
      )

  models_metadata = []
  for task in train_pipeline_job['jobDetail']['taskDetails']:
    if not task['taskName'].startswith('automlforecastingtrainingjob-run'):
      continue
    model_artifact = next(
      artifact
      for artifact in task['outputs']['model']['artifacts']
      if artifact['schemaTitle'] == 'google.VertexModel'
    )
    model_display_name = train_job_to_model_display_name[task['taskName']]
    model_uri = model_artifact['uri']
    models_metadata.append({
      'uri': model_uri,
      'display_name': model_display_name,
    })

  return (json.dumps(models_metadata),)


#### Model_1 Batch Prediction
@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.7.0'],
)
def model_1_predict_job(
    project: str,
    location: str,
    eval_bq_dataset: str,
    models_metadata: str,
    bigquery_source: str,
) -> NamedTuple('Outputs', [
                            ('batch_predict_output_bq_uri', str),
                            ('batch_predict_job_dict', dict)]):

  from google.cloud import aiplatform
  import json

  aiplatform.init(
      project=project,
      location=location,
  )

  models_meta_dict = (
      json.loads(models_metadata)
  )

  model_aip_uri=models_meta_dict[0]["uri"]
  model_aip_uri=model_aip_uri[49:]
  print("Model URI:", model_aip_uri)

  model = aiplatform.Model(model_name=model_aip_uri)
  print("Model dict:", model.to_dict())

  batch_predict_job = model.batch_predict(
      bigquery_source=bigquery_source,
      instances_format="bigquery",
      bigquery_destination_prefix=f'bq://{project}.{eval_bq_dataset}',
      predictions_format="bigquery",
      job_display_name='batch-predict-job-1',
  )

  batch_predict_bq_output_uri = "{}.{}".format(
      batch_predict_job.output_info.bigquery_output_dataset,
      batch_predict_job.output_info.bigquery_output_table)
  
  # if batch_predict_bq_output_uri.startswith("bq://"):
  #   batch_predict_bq_output_uri = batch_predict_bq_output_uri[5:]

  # batch_predict_bq_output_uri.replace(":", ".")

  print(batch_predict_job.to_dict())
  return (batch_predict_bq_output_uri, 
          batch_predict_job.to_dict())

### Model_2 Batch Prediction
@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.7.0'],
)
def model_2_predict_job(
    project: str,
    location: str,
    eval_bq_dataset: str,
    models_metadata: str,
    bigquery_source: str,
) -> NamedTuple('Outputs', [
                            ('batch_predict_output_bq_uri', str),
                            ('batch_predict_job_dict', dict)]):

  from google.cloud import aiplatform
  import json

  aiplatform.init(
      project=project,
      location=location,
  )

  models_meta_dict = (
      json.loads(models_metadata)
  )

  model_aip_uri=models_meta_dict[1]["uri"]
  model_aip_uri=model_aip_uri[49:]
  print("Model URI:", model_aip_uri)

  model = aiplatform.Model(model_name=model_aip_uri)
  print("Model dict:", model.to_dict())

  batch_predict_job = model.batch_predict(
      bigquery_source=bigquery_source,
      instances_format="bigquery",
      bigquery_destination_prefix=f'bq://{project}.{eval_bq_dataset}',
      predictions_format="bigquery",
      job_display_name='batch-predict-job-2',
  )

  batch_predict_bq_output_uri = "{}.{}".format(
      batch_predict_job.output_info.bigquery_output_dataset,
      batch_predict_job.output_info.bigquery_output_table)
  
  # if batch_predict_bq_output_uri.startswith("bq://"):
  #   batch_predict_bq_output_uri = batch_predict_bq_output_uri[5:]

  # batch_predict_bq_output_uri.replace(":", ".")

  print(batch_predict_job.to_dict())
  return (batch_predict_bq_output_uri, 
          batch_predict_job.to_dict())

### Combine Plan Predictions
@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-bigquery==2.18.0'],
)
def create_combined_preds_forecast_table(
  project: str,
  dataset: str,
  model_1_pred_table_uri: str,
  model_2_pred_table_uri: str,
  override: str = 'False',
) -> NamedTuple('Outputs', [('combined_preds_forecast_table_uri', str)]):
  from google.cloud import bigquery
 
  override = bool(override)
  bq_client = bigquery.Client(project=project)
  combined_preds_forecast_table_name = f'{project}.{dataset}.combined_preds_forecast'
  (
    bq_client.query(
      f"""
      CREATE {'OR REPLACE TABLE' if override else 'TABLE IF NOT EXISTS'} 
        `{combined_preds_forecast_table_name}`
      AS (SELECT
          table_a.datetime as datetime, 
          table_a.vertex__timeseries__id, 
          ROUND(table_a.predicted_gross_quantity.value,2) as predicted_gross_quantity_a, 
          ROUND(table_b.predicted_gross_quantity.value, 2) as predicted_gross_quantity_b,
          ROUND((table_a.predicted_gross_quantity.value + table_b.predicted_gross_quantity.value)/2, 2) AS Final_Pred
          FROM
          `{model_1_pred_table_uri[5:]}` AS table_a
          INNER JOIN `{model_2_pred_table_uri[5:]}` AS table_b
              ON table_a.datetime = table_b.datetime
              and table_a.vertex__timeseries__id = table_b.vertex__timeseries__id
              );
          """
    )
    .result()
  )

  return (
    f'bq://{combined_preds_forecast_table_name}',
  )


##### Configure Pipeline
@kfp.v2.dsl.pipeline(
  name=f'{VERSION}-{PIPELINE_TAG}'.replace('_', '-')
)
def pipeline(
    project: str,
    batch_job_location: str,
    pipeline_location: str,
    version: str,
    eval_bq_dataset: str,
    train_pipeline_name: str,
    forecast_products_override: str,
    forecast_products_table_uri: str,
    forecast_activities_override: str,
    forecast_activities_table_uri: str,
    forecast_locations_override: str,
    # forecast_locations_table_uri: str,
    forecast_plan_table_uri: str,
    forecast_plan_override: str,
    forecast_activities_expected_historical_last_date: str,
    context_window: int,
    forecast_horizon: int,
    time_granularity_unit: str,
    time_granularity_quantity: int,
    override: str,):
  
  create_forecast_input_table_specs_op = create_forecast_input_table_specs(
    project=project,
    pipeline_location=pipeline_location,
    train_pipeline_name=train_pipeline_name,
    forecast_products_table_uri=forecast_products_table_uri,
    forecast_activities_table_uri=forecast_activities_table_uri,
    forecast_plan_table_uri=forecast_plan_table_uri,
    # forecast_locations_table_uri=forecast_locations_table_uri,
    # time_granularity_unit=time_granularity_unit,
    # time_granularity_quantity=time_granularity_quantity,
  )

  forecast_validation_op = gcc_aip_forecasting.ForecastingValidationOp(
      input_tables=create_forecast_input_table_specs_op.outputs['forecast_input_table_specs'],
      validation_theme='FORECASTING_PREDICTION',
  )

  forecast_preprocess_op = gcc_aip_forecasting.ForecastingPreprocessingOp(
      project=project,
      input_tables=create_forecast_input_table_specs_op.outputs['forecast_input_table_specs'],
      preprocessing_bigquery_dataset=eval_bq_dataset,
  )
  forecast_preprocess_op.after(forecast_validation_op)

  predict_table_path_op = get_predict_table_path(
    forecast_preprocess_op.outputs['preprocess_metadata'],
  )

  models_metadata_op = get_models_metadata(
    project=project,
    location=pipeline_location,
    train_pipeline_name=train_pipeline_name,
  )

  model_1_predict_job_op = model_1_predict_job(
      project=project,
      location=batch_job_location,
      eval_bq_dataset=eval_bq_dataset,
      models_metadata=models_metadata_op.outputs['models_metadata'],
      bigquery_source=predict_table_path_op.outputs['preprocess_bq_uri'],
  )


  model_2_predict_job_op = model_2_predict_job(
      project=project,
      location=batch_job_location,
      eval_bq_dataset=eval_bq_dataset,
      models_metadata=models_metadata_op.outputs['models_metadata'],
      bigquery_source=predict_table_path_op.outputs['preprocess_bq_uri'],
  )

  create_combined_preds_forecast_table_op = create_combined_preds_forecast_table(
      project=project,
      dataset=eval_bq_dataset,
      model_1_pred_table_uri=model_1_predict_job_op.outputs['batch_predict_output_bq_uri'],
      model_2_pred_table_uri=model_2_predict_job_op.outputs['batch_predict_output_bq_uri'],
      override=override,
  )
  
  
### Compile Pipeline
kfp.v2.compiler.Compiler().compile(
  pipeline_func=pipeline, 
  package_path=FLAGS.JSON_PACKAGE_PATH
#   package_path='custom_container_pipeline_spec.json',
)

### Submit pipeline
if not PIPELINES.get('forecast') or FLAGS.OVERWRITE:
  response = pipeline_client.create_run_from_job_spec(
    job_spec_path=FLAGS.package_json_path # 'custom_container_pipeline_spec.json',
    # service_account=SERVICE_ACCOUNT, # <--- TODO: Uncomment if needed
    parameter_values={
      'project': FLAGS.PROJECT_ID,
      'batch_job_location': FLAGS.BATCH_JOB_LOCATION,
      'pipeline_location': FLAGS.PIPELINE_LOCATION,
      'version': FLAGS.VERSION,
      'eval_bq_dataset': FLAGS.EVAL_DESTINATION_DATASET,
      'train_pipeline_name': FLAGS.TRAIN_PIPELINE_NAME,
      'forecast_products_override': 'False',
      'forecast_products_table_uri': f'bq://{FLAGS.FORECAST_PRODUCTS_TABLE}',
      'forecast_activities_override': 'False',
      'forecast_activities_table_uri': f'bq://{FLAGS.FORECAST_ACTIVITIES_TABLE}',
      'forecast_locations_override': 'False',
      # 'forecast_locations_table_uri': f'bq://{FLAGS.FORECAST_LOCATIONS_TABLE}',
      'forecast_plan_override': 'False',
      'forecast_plan_table_uri': f'bq://{FLAGS.FORECAST_PLAN_TABLE}',
      'forecast_activities_expected_historical_last_date': FLAGS.FORECAST_ACTIVITIES_EXPECTED_HISTORICAL_LAST_DATE,
      'forecast_horizon': FLAGS.FORECAST_HORIZON,
      'time_granularity_unit': FLAGS.TIME_GRANULARITY_UNIT,
      'context_window': FLAGS.HISTORY_WINDOW_n,
      'time_granularity_quantity': FLAGS.TIME_GRANULARITY_QUANTITY,
      'override': 'True'
    },
    pipeline_root=f'{FLAGS.GS_PIPELINE_ROOT_PATH}/{FLAGS.VERSION}', 
  )
  PIPELINES['forecast'] = response['name']
