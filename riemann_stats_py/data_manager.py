import pandas as pd

class DataProcessing:
    """
    Class to handle data loading and preparation.
    """
    @staticmethod
    def load_data(filepath, separator=";", decimal="."):
        """
        Loads data from a CSV file.

        Parameters:
            filepath (str): Path to the CSV file.
            separator (str): Delimiter of the CSV file.
            decimal (str): Decimal character in the CSV file.
        
        Returns:
            pandas.DataFrame: Loaded data.
        """
        return pd.read_csv(filepath, sep=separator, decimal=decimal)
