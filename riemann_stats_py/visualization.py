import matplotlib.pyplot as plt

class Visualization:
    """
    Class for visualizing data and analyses.
    """

    @staticmethod
    def plot_data(data, title="Data Plot", xlabel="X-axis", ylabel="Y-axis"):
        """
        Plots data on a 2D plane.

        Parameters:
            data (pandas.DataFrame): Data to be plotted.
            title (str): Title of the plot.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
        """
        plt.figure(figsize=(10, 8))
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
