import numpy as np
import pandas as pd
from scania.s3_bucket_operations.s3_operations import S3_Operation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class Preprocessor:
    """
    Description :   This class shall  be used to clean and transform the data before training.
    Version     :   1.2
    Revisions   :   moved setup to cloud
    """

    def __init__(self, log_file):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_file = log_file

        self.null_values_file = self.config["null_values_csv_file"]

        self.n_components = self.config["pca_model"]["n_components"]

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.model_utils = Model_Utils()

        self.s3 = S3_Operation()

    def remove_columns(self, data, columns):
        """
        Method Name :   remove_columns
        Description :   This method removes the given columns from a pandas dataframe.
        
        Output      :   A pandas DataFrame after removing the specified columns.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.remove_columns.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        self.data = data

        self.columns = columns

        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)

            self.log_writer.log(
                self.log_file, f"Dropped {columns} from {data}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return self.useful_data

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the features and a Label Coulmns.
        
        Output      :   Returns two separate dataframes, one containing features and the other containing Labels .
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.separate_label_feature.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            self.X = data.drop(labels=label_column_name, axis=1)

            self.Y = data[label_column_name]

            self.log_writer.log(
                self.log_file, f"Separated {label_column_name} from {data}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return self.X, self.Y

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def replace_invalid_values(self, data):
        """
        Method Name :   replace_invalid_values
        Description :   This method replaces the invalid values with np.nan
        
        Output      :   A dataframe without invalid values is returned
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.replace_invalid_values.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.log_file,
            )

            data.replace(to_replace="'na'", value=np.nan, inplace=True)

            self.log_writer.log(self.log_file, "Replaced " "na" " with np.nan")

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def is_null_present(self, data):
        """
        Method Name :   is_null_present
        Description :   This method checks whether there are null values present in the pandas dataframe or not.
        
        Output      :   Returns True if null values are present in the DataFrame, False if they are not present and
                        returns the list of columns for which null values are present.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.is_null_present.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        null_present = False

        cols_with_missing_values = []

        cols = data.columns

        try:
            self.null_counts = data.isna().sum()

            self.log_writer.log(
                self.log_file, f"Null values count is : {self.null_counts}",
            )

            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    null_present = True

                    cols_with_missing_values.append(cols[i])

            self.log_writer.log(
                self.log_file, "created cols with missing values",
            )

            if null_present:
                self.log_writer.log(
                    self.log_file,
                    "null values were found the columns...preparing dataframe with null values",
                )

                self.dataframe_with_null = pd.DataFrame()

                self.dataframe_with_null["columns"] = data.columns

                self.dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.log_writer.log(
                    self.log_file, "Created dataframe with null values",
                )

                self.s3.upload_df_as_csv(
                    self.dataframe_with_null,
                    self.null_values_file,
                    self.input_files_bucket,
                    self.null_values_file,
                    self.log_file,
                )

            else:
                self.log_writer.log(
                    self.log_file,
                    "No null values are present in cols. Skipped the creation of dataframe",
                )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return null_present

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def encode_target_cols(self, data):
        """
        Method Name :   encode_target_cols
        Description :   This method encodes all the categorical values in the training set.
        
        Output      :   A dataframe which has target values encoded.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.encode_target_cols.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.log_file,
            )

            data["class"] = data["class"].map({"'neg'": 0, "'pos'": 1})

            self.log_writer.log(
                self.log_file, "Encoded target cols in dataframe",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def impute_missing_values(self, data):
        """
        Method Name :   impute_missing_values
        Description :   This method replaces all the missing values in the dataframe using mean values of the column.
        
        Output      :   A dataframe which has all the missing values imputed.
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.impute_missing_values.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.log_file,
            )

            data = data[data.columns[data.isnull().mean() < 0.6]]

            data = data.apply(pd.to_numeric)

            for col in data.columns:
                data[col] = data[col].replace(np.NaN, data[col].mean(), inplace=True)

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return data

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def apply_pca_transform(self, X_scaled_data):
        """
        Method Name : apply_pca_transform
        Description : This method applies the PCA transformation the features cols
        
        Output      : A dataframe with scaled values
        On Failure  : Write an exception log and then raise an exception

        Version     : 1.2
        Revisions   : moved setup to cloud
        """
        method_name = self.apply_pca_transform.__name__

        try:
            self.log_writer.start_log(
                "start", self.class_name, method_name, self.log_file,
            )

            pca = PCA(n_components=self.n_components)

            pca_model_name = self.model_utils.get_model_name(pca, self.log_file)

            self.log_writer.log(
                self.log_file,
                f"Initialized {pca_model_name} model with n_components to {self.n_components}",
            )

            new_data = pca.fit_transform(X_scaled_data)

            self.log_writer.log(
                self.log_file, f"Transformed the data using {pca_model_name} model",
            )

            principal_x = pd.DataFrame(new_data, index=self.data.index)

            self.log_writer.log(
                self.log_file, "Created a dataframe for the transformed data",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return principal_x

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def scale_numerical_columns(self, data):
        """
        Method Name : scale_numerical_columns
        Description : This method scales the numerical values using the Standard scaler.
        
        Output      : A dataframe with scaled values
        On Failure  : Write an exception log and then raise an exception

        Version     : 1.2
        Revisions   : moved setup to cloud
        """
        method_name = self.scale_numerical_columns.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            self.data = data

            self.scaler = StandardScaler()

            self.log_writer.log(
                self.log_file, f"Initialized {self.scaler.__class__.__name__}",
            )

            self.scaled_data = self.scaler.fit_transform(self.data)

            self.log_writer.log(
                self.log_file,
                f"Transformed data using {self.scaler.__class__.__name__}",
            )

            self.scaled_num_df = pd.DataFrame(
                data=self.scaled_data, columns=self.data.columns, index=self.data.index
            )

            self.log_writer.log(
                self.log_file, "Converted transformed data to dataframe",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return self.scaled_num_df

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )

    def get_columns_with_zero_std_deviation(self, data):
        """
        Method Name :   get_columns_with_zero_std_deviation
        Description :   This method finds out the columns which have a standard deviation of zero.
        
        Output      :   List of the columns with standard deviation of zero
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_columns_with_zero_std_deviation.__name__

        self.log_writer.start_log(
            "start", self.class_name, method_name, self.log_file,
        )

        try:
            data_n = data.describe()

            cols_to_drop = [x for x in data.columns if data_n[x]["std"] == 0]

            self.log_writer.log(
                self.log_file, "Got cols with zero standard deviation",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file,
            )

            return cols_to_drop

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file,
            )
