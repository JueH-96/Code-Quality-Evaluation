# Read input from stdin
A, B = map(int, input().split())

# Compute A^B + B^A
result = A**B + B**A

# Print the result to stdout
print(result)