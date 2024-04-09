import matplotlib.pyplot as plt
import numpy as np


class FuzzyMembership:
    def __init__(self, name: str, points: list[float]):
        """
        Create a fuzzy membership function with the given points.
        :param name: Name of the membership function
        :param points: List of 3 points that define the triangular membership function
        """
        self.name = name
        self.points = points

    def __call__(self, x):
        """
        Create a triangular membership function with the given points.
        :param x: Value to calculate the membership function for
        :return: Membership function
        """
        if x <= self.points[0]:
            return 1 if self.points[0] == self.points[1] else 0
        elif self.points[0] < x <= self.points[1]:
            return (x - self.points[0]) / (self.points[1] - self.points[0])
        elif self.points[1] < x <= self.points[2]:
            return (self.points[2] - x) / (self.points[2] - self.points[1])
        else:
            return 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FuzzyVariable:
    def __init__(self, name: str, membership_functions: list[FuzzyMembership] = None):
        """
        Create a fuzzy variable with the given membership functions.
        :param name: Name of the fuzzy variable
        :param membership_functions: List of membership functions
        """
        self.name = name
        self.membership_functions = membership_functions if membership_functions is not None else []

    def add_membership_function(self, membership_function):
        """
        Add a membership function to the fuzzy variable.
        :param membership_function: Membership function to add
        """
        self.membership_functions.append(membership_function)

    def get_membership_function(self, name):
        """
        Get the membership function with the given name.
        :param name: Name of the membership function to get
        :return: Membership function with the given name
        """
        for mf in self.membership_functions:
            if mf.name == name:
                return mf
        raise ValueError(f'Membership function with name "{name}" not found')

    def delete_membership_function(self, name):
        """
        Delete the membership function with the given name.
        :param name: Name of the membership function to delete
        """
        self.membership_functions.remove(self.get_membership_function(name))

    def visualize_membership(self):
        """
        Visualize the membership functions of the fuzzy variable.
        """
        # Define the range of x values
        x = np.linspace(self.membership_functions[0].points[0], self.membership_functions[-1].points[-1], 101)
        # Plot the membership functions
        plt.figure(figsize=(10, 5))
        # Create a plot for each membership function
        for f in self.membership_functions:
            y = [f(i) for i in x]
            plt.plot(x, y, label=f.name)
        plt.title('Membership Functions')
        plt.xlabel('Value')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# Define the membership functions
food_quality = FuzzyVariable('Food Quality')
food_quality.add_membership_function(FuzzyMembership('Low food', [0, 0, 5]))
food_quality.add_membership_function(FuzzyMembership('Medium food', [0, 5, 10]))
food_quality.add_membership_function(FuzzyMembership('High food', [5, 10, 10]))
food_quality.visualize_membership()

service_quality = FuzzyVariable('Service Quality')
service_quality.add_membership_function(FuzzyMembership('Low service', [0, 0, 5]))
service_quality.add_membership_function(FuzzyMembership('Medium service', [0, 5, 10]))
service_quality.add_membership_function(FuzzyMembership('High service', [5, 10, 10]))
service_quality.visualize_membership()

tip_amount = FuzzyVariable('Tip Amount')
tip_amount.add_membership_function(FuzzyMembership('Low tip', [0, 0, 13]))
tip_amount.add_membership_function(FuzzyMembership('Medium tip', [0, 13, 25]))
tip_amount.add_membership_function(FuzzyMembership('High tip', [13, 25, 25]))
tip_amount.visualize_membership()
