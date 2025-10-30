# Read the input from stdin
x = float(input())

# Classify the body temperature
if x >= 38.0:
    print(1)  # High fever
elif x >= 37.5:
    print(2)  # Fever
else:
    print(3)  # Normal