# Read the input values
A, B = map(int, input().split())

# Calculate A^B + B^A and print the result
print(A ** B + B ** A)