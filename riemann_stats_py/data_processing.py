import pandas as pd


class DataProcessing:
    """
    Class for handling data loading and preparation.

    This class offers static methods to read CSV files and convert them into DataFrames,
    facilitating data analysis in data science projects.
    """

    @staticmethod
    def load_data(filepath: str, separator: str = ";", decimal: str = ".") -> pd.DataFrame:
        """
        Loads data from a CSV file and returns a DataFrame.

        This method reads a CSV file using the pandas library. It allows specifying the delimiter
        and the decimal character, with default values provided for ease of use.

        Parameters:
            filepath (str): Path to the CSV file.
            separator (str, optional): Delimiter used in the CSV file. Defaults to ";".
            decimal (str, optional): Character used for decimal points in the CSV file. Defaults to ".".

        Returns:
            pandas.DataFrame: A DataFrame containing the data loaded from the CSV file.
        """
        return pd.read_csv(filepath, sep=separator, decimal=decimal)
