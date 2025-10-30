# Read the body temperature from stdin
X = float(input())

# Classify the body temperature
if X >= 38.0:
    print(1)  # High fever
elif X >= 37.5:
    print(2)  # Fever
else:
    print(3)  # Normal