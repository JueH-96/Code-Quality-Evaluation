# YOUR CODE HERE
X_str = input().strip()

# Remove trailing zeros and the decimal point if it's the last character
X_str = X_str.rstrip('0').rstrip('.') if '.' in X_str else X_str

print(X_str)