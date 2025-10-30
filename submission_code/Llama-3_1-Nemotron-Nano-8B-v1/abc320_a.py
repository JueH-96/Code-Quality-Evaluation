# Read the input values
A, B = map(int, input().split())

# Calculate A^B and B^A
result = A ** B + B ** A

# Output the result
print(result)