# Read the input from stdin
A, B = map(int, input().split())

# Calculate A^B and B^A
A_to_B = A ** B
B_to_A = B ** A

# Calculate the sum
result = A_to_B + B_to_A

# Print the result
print(result)