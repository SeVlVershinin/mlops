import io
import json
import pickle
from datetime import timedelta, datetime
from typing import Any, Dict

import pandas as pd
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


AUTHOR = "sergey-vershinin"
BUCKET = Variable.get("S3_BUCKET")
S3_HOOK = S3Hook("s3_connection")

DEFAULT_ARGS = {
    'owner': AUTHOR,
    'email': 'to_sergey@list.ru',
    'email_on_failure': True,
    'email_on_retry': False,
    'retry': 3,
    'retry-delay': timedelta(minutes=1)
}


models = dict(
    zip(["random_forest", "linear_regression", "decision_tree"],
        [RandomForestRegressor, LinearRegression, DecisionTreeRegressor,]))

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude",]
TARGET = "MedHouseVal"


def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_dag(dag_id: str, model_name: str):

    ####### ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ШАГОВ DAGа #######
    def save_dataset_to_s3(dataset, dataset_name) -> int:
        """
        Сохраняет датасет любого вида на S3 под указанным именем в формате pickle
        :return: количество байт в сохраненном файле
        """
        buffer = io.BytesIO()
        pickle.dump(dataset, buffer)
        buffer.seek(0)
        n_bytes = buffer.getbuffer().nbytes
        S3_HOOK.load_file_obj(
            file_obj=buffer,
            key=f"{AUTHOR}/{model_name}/datasets/{dataset_name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        return n_bytes

    def load_dataset_from_s3(dataset_name) -> Any:
        """
        Загружает pickle-файл с датасетом из с S3 и возвращает полученный объект
        """
        s3_object = S3_HOOK.get_key(
            key=f"{AUTHOR}/{model_name}/datasets/{dataset_name}.pkl",
            bucket_name=BUCKET
        )
        buffer = io.BytesIO(s3_object.get()['Body'].read())
        dataset = pickle.load(buffer)
        return dataset

    def save_dict_as_json_to_s3(data_dict, file_name):
        """
        Сохраняет словарь в формате json на S3
        """
        buffer = io.BytesIO()
        buffer.write(json.dumps(data_dict).encode())
        buffer.seek(0)
        S3_HOOK.load_file_obj(
            file_obj=buffer,
            key=f"{AUTHOR}/{model_name}/results/{file_name}.json",
            bucket_name=BUCKET,
            replace=True
        )

    ####### ШАГИ DAGа #######
    def init() -> Dict[str, Any]:
        metrics = {
            "model_name": model_name,
            "pipeline_started_At": get_current_timestamp(),
        }
        return metrics


    def get_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='init')

        metrics["get_dataset_started_at"] = get_current_timestamp()

        # получим датасет California housing, объединим фичи и таргет в один dataframe и сохраним его на S3
        housing_dataset = fetch_california_housing(as_frame=True)
        data = pd.concat([housing_dataset["data"], pd.DataFrame(housing_dataset["target"])], axis=1)
        bytes_saved = save_dataset_to_s3(data, 'california_housing')

        metrics["get_dataset_finished_at"] = get_current_timestamp()
        metrics["dataset_size"] = {
            "number_of_rows": data.shape[0],
            "number_of_columns": data.shape[1],
            "number_of_bytes": bytes_saved
        }
        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='get_data')
        metrics["prepare_data_started_at"] = get_current_timestamp()

        data = load_dataset_from_s3('california_housing')
        X_train, X_test, y_train, y_test = train_test_split(
            data[FEATURES],
            data[TARGET],
            test_size=0.3,
            random_state=42
        )
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        save_dataset_to_s3((X_train_fitted, X_test_fitted, y_train, y_test), 'train_test_scaled')

        metrics["prepare_data_finished_at"] = get_current_timestamp()
        metrics["feature_names"] = FEATURES
        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='prepare_data')
        metrics["train_model_started_at"] = get_current_timestamp()

        X_train, X_test, y_train, y_test = load_dataset_from_s3('train_test_scaled')

        model = models[model_name]()
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        rmse = root_mean_squared_error(y_test, prediction)

        metrics["train_model_finished_At"] = get_current_timestamp()
        metrics["model_metrics"] = {
            "r2 score": r2,
            "rmse score": rmse
        }
        return metrics

    def save_results(**kwargs) -> None:
        metrics = kwargs['ti'].xcom_pull(task_ids='train_model')
        save_dict_as_json_to_s3(metrics, 'metrics')


    ####### ИНИЦИАЛИЗАЦИЯ DAGa #######
    dag = DAG(
        dag_id=dag_id,
        schedule_interval='0 1 * * *',
        start_date=days_ago(2),
        catchup=False,
        tags=['mlops'],
        default_args=DEFAULT_ARGS
    )

    with dag:
        task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)
        task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag, provide_context=True)
        task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)
        task_train_model = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag, provide_context=True)
        task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"{AUTHOR}_{model_name}", model_name)