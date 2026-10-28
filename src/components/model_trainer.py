
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, train_array, test_array, processor_path=None
    ):
        try:
            logging.info(
                "Splitting training and testing input data"
            )

            # Separate features (X) and target (y)
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define regression models
            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(
                    random_state=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                ),
            }

            # Train and evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            if not model_report:
                raise ValueError(
                    "Model evaluation returned no results."
                )

            # Find the model with the highest R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}, "
                f"R2 score: {best_model_score}"
            )

            # Reject the model if its R2 score is too low
            if best_model_score < 0.6:
                raise ValueError(
                    f"No suitable model found. "
                    f"Best R2 score: {best_model_score:.4f}"
                )

            # Create the artifacts folder if necessary
            os.makedirs("artifacts", exist_ok=True)

            # Save the best trained model
            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model,
            )

            # Evaluate the best model on test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(
                f"Test R2 score: {r2_square}"
            )

            return r2_square

        except Exception as e:
            logging.exception(
                "Error occurred during model training"
            )
            raise CustomException(e, sys) from e
