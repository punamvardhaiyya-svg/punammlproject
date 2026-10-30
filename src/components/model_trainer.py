
# Import required libraries
import os
import sys
from dataclasses import dataclass

# Import regression models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Import evaluation metric
from sklearn.metrics import r2_score

# Import XGBoost
from xgboost import XGBRegressor

# Import custom project utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:

    # Path where best model will be saved
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        # Initialize model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, train_array, test_array, processor_path=None
    ):

        try:

            logging.info("Starting model training")

            # Step 1: Separate input features and target variable

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Step 2: Define regression models

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
                    random_state=42,
                    objective="reg:squarederror",
                    verbosity=0
                ),

                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                )
            }

            # Step 3: Define hyperparameters

            param_grids = {

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 5]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },

                "XGBoost Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                }
            }

            # Step 4: Create dictionaries to store results

            model_report = {}
            trained_models = {}
            best_params = {}

            # Step 5: Apply hyperparameter tuning

            for model_name, model in models.items():

                logging.info(
                    f"Hyperparameter tuning started for {model_name}"
                )

                grid_search = GridSearchCV(

                    estimator=model,

                    param_grid=param_grids[model_name],

                    scoring="r2",

                    cv=5,

                    n_jobs=-1,

                    verbose=1

                )

                # Train model with different parameter combinations

                grid_search.fit(X_train, y_train)

                # Get best tuned model

                best_model = grid_search.best_estimator_

                # Store best model

                trained_models[model_name] = best_model

                # Store best cross-validation score

                model_report[model_name] = grid_search.best_score_

                # Store best parameters

                best_params[model_name] = grid_search.best_params_

                logging.info(
                    f"Best parameters for {model_name}: "
                    f"{grid_search.best_params_}"
                )

                logging.info(
                    f"Best CV R2 score: {grid_search.best_score_}"
                )

            # Step 6: Select best model

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = trained_models[best_model_name]

            logging.info(
                f"Best model selected: {best_model_name}"
            )

            logging.info(
                f"Best hyperparameters: {best_params[best_model_name]}"
            )

            # Step 7: Predict on test data

            predicted = best_model.predict(X_test)

            # Step 8: Calculate final R2 score

            r2_square = r2_score(y_test, predicted)

            logging.info(
                f"Final Test R2 Score: {r2_square}"
            )

            # Step 9: Check model performance

            if r2_square < 0.6:

                raise ValueError(
                    f"Model performance is low. "
                    f"R2 Score: {r2_square:.4f}"
                )

            # Step 10: Create artifacts folder

            os.makedirs("artifacts", exist_ok=True)

            # Step 11: Save best tuned model

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best hyperparameter tuned model saved successfully"
            )

            # Return test R2 score

            return r2_square

        except Exception as e:

            logging.exception(
                "Error occurred during model training"
            )

            raise CustomException(e, sys) from e
