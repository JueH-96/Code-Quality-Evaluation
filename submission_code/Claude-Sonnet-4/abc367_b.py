# YOUR CODE HERE
X = input().strip()
# Convert to float and back to string to normalize
num = float(X)
# Format as string and remove trailing zeros and decimal point
result = f"{num:.3f}".rstrip('0').rstrip('.')
print(result)