# Read the input values
A, B = map(int, input().split())

# Compute A^B and B^A
result = (A ** B) + (B ** A)

# Print the result
print(result)