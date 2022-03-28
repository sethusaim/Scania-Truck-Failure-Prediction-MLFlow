from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class Model_Finder:
    """
    Description :   This method is used for hyperparameter tuning of selected models
                    some preprocessing steps and then train the models and register them in mlflow
    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.log_writer = App_Logger()

        self.model_utils = Model_Utils()

        self.ada_model = AdaBoostClassifier()

        self.rf_model = RandomForestClassifier()

    def get_best_model_for_adaboost(self, train_x, train_y):
        """
        Method Name :   get_best_model_for_adaboost
        Description :   get the parameters for AdaBoost Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_model_for_adaboost.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            self.ada_model_name = self.model_utils.get_model_name(
                self.ada_model, self.log_file
            )

            self.adaboost_best_params = self.model_utils.get_model_params(
                self.ada_model, "adaboost_model", train_x, train_y, self.log_file,
            )

            self.n_estimators = self.adaboost_best_params["n_estimators"]

            self.learning_rate = self.adaboost_best_params["learning_rate"]

            self.random_state = self.adaboost_best_params["random_state"]

            self.log_writer.log(
                self.log_file,
                f"{self.ada_model_name} model best params are {self.adaboost_best_params}",
            )

            self.ada_model = AdaBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )

            self.log_writer.log(
                self.log_file,
                f"Initialized {self.ada_model_name} with {self.adaboost_best_params} as params",
            )

            self.ada_model.fit(train_x, train_y)

            self.log_writer.log(
                self.log_file,
                f"Created {self.ada_model_name} based on the {self.adaboost_best_params} as params",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return self.ada_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def get_best_model_for_rf(self, train_x, train_y):
        """
        Method Name :   get_best_model_for_rf
        Description :   get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_model_for_rf.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            self.rf_model_name = self.rf_model.__class__.__name__

            self.rf_best_params = self.model_utils.get_model_params(
                self.rf_model, "rf_model", train_x, train_y, self.log_file,
            )

            self.criterion = self.rf_best_params["criterion"]

            self.max_depth = self.rf_best_params["max_depth"]

            self.max_features = self.rf_best_params["max_features"]

            self.n_estimators = self.rf_best_params["n_estimators"]

            self.log_writer.log(
                self.log_file,
                f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            self.rf_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
            )

            self.log_writer.log(
                self.log_file,
                f"Initialized {self.rf_model_name} with {self.rf_best_params} as params",
            )

            self.rf_model.fit(train_x, train_y)

            self.log_writer.log(
                self.log_file,
                f"Created {self.rf_model_name} based on the {self.rf_best_params} as params",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return self.rf_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   Find out the Model which has the best score.
        
        Output      :   The best model name and the model object
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.0
        Revisions   :   None
        """
        method_name = self.get_trained_models.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            ada_model = self.get_best_model_for_adaboost(
                train_x=train_x, train_y=train_y
            )

            ada_model_score = self.model_utils.get_model_score(
                ada_model, test_x, test_y, self.log_file,
            )

            rf_model = self.get_best_model_for_rf(train_x=train_x, train_y=train_y)

            rf_model_score = self.model_utils.get_model_score(
                rf_model, test_x, test_y, self.log_file,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return (
                rf_model,
                rf_model_score,
                ada_model,
                ada_model_score,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )
