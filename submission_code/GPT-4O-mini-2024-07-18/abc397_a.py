# YOUR CODE HERE
def classify_temperature(X):
    if X >= 38.0:
        return 1  # High fever
    elif X >= 37.5:
        return 2  # Fever
    else:
        return 3  # Normal

# Read input
X = float(input().strip())
# Get classification
result = classify_temperature(X)
# Print the result
print(result)