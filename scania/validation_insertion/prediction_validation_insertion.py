from scania.data_transform.data_transformation_pred import Data_Transform_Pred
from scania.data_type_valid.data_type_valid_pred import DB_Operation_Pred
from scania.raw_data_validation.pred_data_validation import Raw_Pred_Data_Validation
from utils.logger import App_Logger
from utils.read_params import read_params


class Pred_Validation:
    """
    Description :   This class is used for validating all the Prediction batch files
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, bucket):
        self.raw_data = Raw_Pred_Data_Validation(bucket)

        self.data_transform = Data_Transform_Pred()

        self.db_operation = DB_Operation_Pred()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.pred_main_log = self.config["pred_db_log"]["pred_main"]

        self.good_data_db_name = self.config["mongodb"]["scania_data_db_name"]

        self.good_data_collection_name = self.config["mongodb"][
            "scania_pred_data_collection"
        ]

        self.log_writer = App_Logger()

    def prediction_validation(self):
        """
        Method Name :   prediction_validation
        Description :   This method is responsible for converting raw data to cleaned data for prediction

        Output      :   Raw data is converted to cleaned data for prediction
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.prediction_validation.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.pred_main_log,
            )

            (
                LengthOfDateStampInFile,
                LengthOfTimeStampInFile,
                column_names,
                noofcolumns,
            ) = self.raw_data.values_from_schema()

            regex = self.raw_data.get_regex_pattern()

            self.raw_data.validate_raw_fname(
                regex, LengthOfDateStampInFile, LengthOfTimeStampInFile
            )

            self.raw_data.validate_col_length(NumberofColumns=noofcolumns)

            self.raw_data.validate_missing_values_in_col()

            self.log_writer.log(
                self.pred_main_log, "Raw Data Validation Completed !!",
            )

            self.log_writer.log(
                self.pred_main_log, "Starting Data Transformation",
            )

            self.data_transform.add_quotes_to_string()

            self.log_writer.log(
                self.pred_main_log, "Data Transformation completed !!",
            )

            self.db_operation.insert_good_data_as_record(
                db_name=self.good_data_db_name,
                collection_name=self.good_data_collection_name,
            )

            self.log_writer.log(
                self.pred_main_log, "Data type validation Operation completed !!",
            )

            self.db_operation.export_collection_to_csv(
                db_name=self.good_data_db_name,
                collection_name=self.good_data_collection_name,
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.pred_main_log,
            )

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.pred_main_log,
            )
