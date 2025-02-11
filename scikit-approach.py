import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the universe of discourse
food_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'Food Quality')
service_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'Service Quality')
tip_amount = ctrl.Consequent(np.arange(0, 26, 1), 'Tip Amount')

# Define the membership functions
food_quality['Low'] = fuzz.trimf(food_quality.universe, [0, 0, 5])
food_quality['Medium'] = fuzz.trimf(food_quality.universe, [0, 5, 10])
food_quality['High'] = fuzz.trimf(food_quality.universe, [5, 10, 10])

service_quality['Low'] = fuzz.trimf(service_quality.universe, [0, 0, 5])
service_quality['Medium'] = fuzz.trimf(service_quality.universe, [0, 5, 10])
service_quality['High'] = fuzz.trimf(service_quality.universe, [5, 10, 10])

tip_amount['Low'] = fuzz.trimf(tip_amount.universe, [0, 0, 13])
tip_amount['Medium'] = fuzz.trimf(tip_amount.universe, [0, 13, 25])
tip_amount['High'] = fuzz.trimf(tip_amount.universe, [13, 25, 25])

# Define the rules
rule1 = ctrl.Rule(food_quality['Low'] | service_quality['Low'], tip_amount['Low'])
rule2 = ctrl.Rule(service_quality['Medium'], tip_amount['Medium'])
rule3 = ctrl.Rule(food_quality['High'] | service_quality['High'], tip_amount['High'])

# Create the control system
tip_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Create the inference system
tipping = ctrl.ControlSystemSimulation(tip_ctrl)

# Input the crisp values
tipping.input['Food Quality'] = 10
tipping.input['Service Quality'] = 10

# Compute the output
tipping.compute()

print(tipping.output['Tip Amount'])