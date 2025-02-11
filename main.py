import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any

OR = 'OR'
AND = 'AND'
NONE = 'NONE'


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
        if membership_function.name in self.membership_functions:
            raise ValueError(f'Membership function with name "{membership_function.name}" already exists')
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
        if name not in self.membership_functions:
            raise ValueError(f'Membership function with name "{name}" not found')
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


class FuzzyRule:
    def __init__(self, antecedent: list[tuple[str, str]], operators: list[str], consequent: Callable[[float, float], float]):
        """
        Create a fuzzy rule with the given antecedent and consequent.
        :param antecedent: Antecedent of the rule
        :param operators: Operators of the rule
        :param consequent: Consequent of the rule which is the function
        """
        self.antecedent = antecedent
        self.operators = operators
        self.consequent = consequent


class FuzzySystem:
    def __init__(self, fuzzy_variables: list[FuzzyVariable] = None, fuzzy_rules: list[FuzzyRule] = None):
        """
        Create a fuzzy system with the given fuzzy variables and rules.
        :param fuzzy_variables: List of fuzzy variables
        :param fuzzy_rules: List of fuzzy rules
        """
        self.fuzzy_variables = fuzzy_variables if fuzzy_variables is not None else []
        self.fuzzy_rules = fuzzy_rules if fuzzy_rules is not None else []

    def add_fuzzy_variable(self, fuzzy_variable):
        """
        Add a fuzzy variable to the fuzzy system.
        :param fuzzy_variable: Fuzzy variable to add
        """
        if fuzzy_variable.name in self.fuzzy_variables:
            raise ValueError(f'Fuzzy variable with name "{fuzzy_variable.name}" already exists')
        self.fuzzy_variables.append(fuzzy_variable)

    def get_fuzzy_variable(self, name):
        """
        Get the fuzzy variable with the given name.
        :param name: Name of the fuzzy variable to get
        :return: Fuzzy variable with the given name
        """
        for fv in self.fuzzy_variables:
            if fv.name == name:
                return fv
        raise ValueError(f'Fuzzy variable with name "{name}" not found')

    def delete_fuzzy_variable(self, name):
        """
        Delete the fuzzy variable with the given name.
        :param name: Name of the fuzzy variable to delete
        """
        if name not in self.fuzzy_variables:
            raise ValueError(f'Fuzzy variable with name "{name}" not found')
        self.fuzzy_variables.remove(self.get_fuzzy_variable(name))

    def infer(self, values: dict[str, float]) -> float:
        """
        Infer the output of the fuzzy system with the given inputs.
        """
        outputs = []
        weights = []
        for rule in self.fuzzy_rules:
            membership_degrees = []
            arguments = []
            for antecedent in rule.antecedent:
                # Get the argument for the consequent function
                arguments.append(values[antecedent[0]])

                # Get the membership function of the fuzzy variable
                mf = self.get_fuzzy_variable(antecedent[0]).get_membership_function(antecedent[1])
                # Calculate the membership degree
                membership_degrees.append(mf(values[antecedent[0]]))

            # print('Adding weight:', weight)
            if len(rule.operators) == 0:
                print('No operator, adding weight of single membership degree:', membership_degrees[0])
                weights.append(membership_degrees[0])
            elif rule.operators[0] == OR:
                print('Operator OR, adding max of membership degrees:', membership_degrees, 'with max:', np.max(membership_degrees))
                weights.append(np.max(membership_degrees))
            else:
                print('Operator AND, adding min of membership degrees:', membership_degrees, 'with min:', np.min(membership_degrees))
                weights.append(np.min(membership_degrees))
            print('Adding output of consequent method with arguments:', arguments, 'and output', rule.consequent(*arguments))
            outputs.append(rule.consequent(*arguments))

        return np.dot(outputs, weights) / sum(weights)


# Define the membership functions
food_quality = FuzzyVariable('Food Quality')
food_quality.add_membership_function(FuzzyMembership('Low', [0, 0, 5]))
food_quality.add_membership_function(FuzzyMembership('Medium', [0, 5, 10]))
food_quality.add_membership_function(FuzzyMembership('High', [5, 10, 10]))
# food_quality.visualize_membership()

service_quality = FuzzyVariable('Service Quality')
service_quality.add_membership_function(FuzzyMembership('Low', [0, 0, 5]))
service_quality.add_membership_function(FuzzyMembership('Medium', [0, 5, 10]))
service_quality.add_membership_function(FuzzyMembership('High', [5, 10, 10]))
# service_quality.visualize_membership()

tip_amount = FuzzyVariable('Tip Amount')
tip_amount.add_membership_function(FuzzyMembership('Low', [0, 0, 13]))
tip_amount.add_membership_function(FuzzyMembership('Medium', [0, 13, 25]))
tip_amount.add_membership_function(FuzzyMembership('High', [13, 25, 25]))
# tip_amount.visualize_membership()


# Define the rules output methods
def rule1_3(x1: float, x2: float) -> float:
    return 5/4*(x1 + x2)


def rule2(x1: float) -> float:
    return 5/2*x1


# Define the rules
rules = [
    FuzzyRule([(food_quality.name, 'Low'), (service_quality.name, 'Low')], [OR], rule1_3),
    FuzzyRule([(service_quality.name, 'Medium')], [], rule2),
    FuzzyRule([(food_quality.name, 'High'), (service_quality.name, 'High')], [OR], rule1_3)
]

# Create the fuzzy system
fs = FuzzySystem([food_quality, service_quality, tip_amount], rules)

values = {
    food_quality.name: 10,
    service_quality.name: 10
}

print(fs.infer(values))
