import mlflow
import pandas as pd
from pandas import Series, DataFrame
from mlflow.models import infer_signature

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


AUTHOR_NAME = "sergey_vershinin"
AUTHOR_TG_NICKNAME = "to_sergeyV"

def main():
    run_experiment(
        experiment_id=get_or_create_experiment(experiment_name=AUTHOR_NAME),
        run_name=AUTHOR_TG_NICKNAME,
        data=get_and_prepare_data(),
        models=dict(
            zip(["random_forest", "linear_regression", "decision_tree"],
                [RandomForestRegressor, LinearRegression, DecisionTreeRegressor]))
    )

def run_experiment(
        experiment_id: str,
        run_name: str,
        data: tuple[DataFrame, DataFrame, Series, Series],
        models: dict[str, callable]
) -> None:
    """
    Запускает эксперимент с использованием MLFlow.
    :param experiment_id: ИД эксперимента
    :param run_name: Имя запуска
    :param data: Кортеж с данными для обучения и тестирования
    :param models: Словарь с моделями
    """
    X_train, X_test, y_train, y_test = data
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, description="parent") as parent_run:
        for model_name in models.keys():
            with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:

                # обучаем модель и делаем предсказания
                model = models[model_name]()
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                # сохраняем модель и метрики с помощью MLFlow.
                signature = infer_signature(X_test, prediction)
                model_info = mlflow.sklearn.log_model(model, "sklearn_models", signature=signature)
                mlflow.evaluate(
                    model=model_info.model_uri,
                    data=pd.concat([X_test, y_test], axis=1),
                    targets=y_test.name,
                    model_type="regressor",
                    evaluators=["default"],
                )

def get_or_create_experiment(experiment_name: str) -> str:
    """
    Выполняет поиск эксперимента по имени. Если эксперимент не найден, создает новый.
    :param experiment_name: Имя эксперимента
    :return: ИД эксперимента
    """
    experiments = mlflow.search_experiments(
        filter_string=f"name = '{experiment_name}'"
    )
    return experiments[0].experiment_id if experiments else mlflow.create_experiment(AUTHOR_NAME)


def get_and_prepare_data() -> tuple[DataFrame, DataFrame, Series, Series]:
    """
    Загружает датасет california_housing, разбивает его на тренировочную и тестовую выборки, а также
    выполняет стандартизацию признаков.
    :return: X_train, X_test, y_train, y_test
    """
    # загружаем датасет и делаем разбиение на тренировочную и тестовую выборки
    housing_dataset = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing_dataset["data"],
        housing_dataset["target"],
        test_size=0.3,
        random_state=42
    )

    # стандартизируем признаки и приводим к DataFrame
    scaler = StandardScaler()
    X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_fitted = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # сбрасываем индексы у y_train и y_test, чтобы в дальнейшем не было проблем с конкатенацией
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train_fitted, X_test_fitted, y_train, y_test

main()

