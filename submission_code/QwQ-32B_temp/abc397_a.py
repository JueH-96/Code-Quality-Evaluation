# Read the input as a float
X = float(input().strip())

# Determine the classification based on the conditions
if X >= 38.0:
    print(1)
elif X >= 37.5:
    print(2)
else:
    print(3)