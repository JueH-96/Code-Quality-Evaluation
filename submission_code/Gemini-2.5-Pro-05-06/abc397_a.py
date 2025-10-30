# Read the input temperature X
X = float(input())

# Classify the temperature based on the rules
if X >= 38.0:
    # Higher than or equal to 38.0 °C: "High fever"
    print(1)
elif X >= 37.5:
    # Higher than or equal to 37.5 °C and lower than 38.0 °C: "Fever"
    # This elif branch is only taken if X < 38.0, so X >= 37.5 implies 37.5 <= X < 38.0
    print(2)
else:
    # Lower than 37.5 °C: "Normal"
    # This else branch is only taken if X < 37.5
    print(3)