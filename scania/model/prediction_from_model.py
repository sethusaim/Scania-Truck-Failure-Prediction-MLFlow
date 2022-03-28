import pandas as pd
from botocore.exceptions import ClientError
from scania.data_ingestion.data_loader_prediction import Data_Getter_Pred
from scania.data_preprocessing.preprocessing import Preprocessor
from scania.s3_bucket_operations.s3_operations import S3_Operation
from utils.logger import App_Logger
from utils.read_params import read_params


class Prediction:
    """
    Description :   This class shall be used for loading the production model

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self):
        self.config = read_params()

        self.pred_log = self.config["pred_db_log"]["pred_main"]

        self.model_bucket = self.config["s3_bucket"]["scania_model_bucket"]

        self.input_files_bucket = self.config["s3_bucket"]["inputs_files_bucket"]

        self.prod_model_dir = self.config["models_dir"]["prod"]

        self.pred_output_file = self.config["pred_output_file"]

        self.log_writer = App_Logger()

        self.s3 = S3_Operation()

        self.data_getter_pred = Data_Getter_Pred(self.pred_log)

        self.preprocessor = Preprocessor(self.pred_log)

        self.class_name = self.__class__.__name__

    def delete_pred_file(self, log_file):
        """
        Method Name :   delete_pred_file
        Description :   This method deletes the existing prediction file for the model prediction starts
        
        Output      :   An existing prediction file is deleted
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.delete_pred_file.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, log_file,
        )

        try:
            self.s3.load_object(
                self.pred_output_file, self.input_files_bucket, log_file,
            )

            self.log_writer.log(
                log_file, f"Found existing Prediction batch file. Deleting it.",
            )

            self.s3.delete_file(
                self.pred_output_file, self.input_files_bucket, log_file,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, log_file,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pass

            else:
                self.log_writer.exception_log(
                    e, self.class_name, method_name, log_file,
                )

    def find_correct_model_file(self, cluster_number, bucket, log_file):
        """
        Method Name :   find_correct_model_file
        Description :   This method gets correct model file based on cluster number during prediction
        Output      :   A correct model file is found 
        On Failure  :   Write an exception log and then raise an exception
        Version     :   1.2
        
        Revisions   :   moved setup to cloud
        """
        method_name = self.find_correct_model_file.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, log_file,
        )

        try:
            list_of_files = self.s3.get_files_from_folder(
                bucket, self.prod_model_dir, log_file,
            )

            for file in list_of_files:
                try:
                    if file.index(str(cluster_number)) != -1:
                        model_name = file

                except:
                    continue

            model_name = model_name.split(".")[0]

            self.log_writer.log(
                log_file,
                f"Got {model_name} from {self.prod_model_dir} folder in {bucket} bucket",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, log_file,
            )

            return model_name

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, log_file,
            )

    def predict_from_model(self):
        """
        Method Name :   predict_from_model
        Description :   This method is used for loading from prod model dir of s3 bucket and use them for prediction

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.predict_from_model.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.pred_log,
        )

        try:
            self.s3.delete_pred_file(self.pred_log)

            data = self.data_getter_pred.get_data()

            data = self.preprocessor.replace_invalid_values(data=data)

            is_null_present = self.preprocessor.is_null_present(data=data)

            if is_null_present:
                data = self.preprocessor.impute_missing_values(data=data)

            cols_to_drop = self.preprocessor.get_columns_with_zero_std_deviation(
                data=data
            )

            X = self.preprocessor.remove_columns(data, cols_to_drop)

            X = self.preprocessor.scale_numerical_columns(data=X)

            X = self.preprocessor.apply_pca_transform(X_scaled_data=X)

            kmeans_model_name = self.prod_model_dir + "/" + "KMeans"

            kmeans_model = self.s3.load_model(
                self.model_bucket, kmeans_model_name, self.pred_log,
            )

            clusters = kmeans_model.predict(data)

            data["clusters"] = clusters

            unique_clusters = data["clusters"].unique()

            for i in unique_clusters:
                cluster_data = data[data["clusters"] == i]

                cluster_data = cluster_data.drop(["clusters"], axis=1)

                model_name = self.find_correct_model_file(
                    i, self.model_bucket, self.pred_log,
                )

                prod_model_name = self.prod_model_dir + "/" + model_name

                model = self.s3.load_model(
                    self.model_bucket, prod_model_name, self.pred_log,
                )

                result = list(model.predict(cluster_data))

                result = pd.DataFrame(result, columns=["Predictions"])

                result["Predictions"] = result["Predictions"].map({0: "neg", 1: "pos"})

                self.s3.upload_df_as_csv(
                    result,
                    self.pred_output_file,
                    self.input_files_bucket,
                    self.pred_output_file,
                    self.pred_log,
                )

            self.log_writer.log(self.pred_log, "End of prediction")

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.pred_log,
            )

            return (
                self.input_files_bucket,
                self.pred_output_file,
                result.head().to_json(orient="records"),
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.pred_log,
            )
