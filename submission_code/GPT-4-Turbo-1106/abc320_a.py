# Read input from stdin
A, B = map(int, input().split())

# Calculate A^B + B^A
result = A**B + B**A

# Print the result to stdout
print(result)