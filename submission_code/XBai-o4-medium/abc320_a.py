# Read input values
A, B = map(int, input().split())

# Calculate A^B and B^A, then sum them
result = (A ** B) + (B ** A)

# Print the result
print(result)