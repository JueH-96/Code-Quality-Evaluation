# Read the input from stdin
A, B = map(int, input().split())

# Calculate A^B and B^A
a_to_b = A ** B
b_to_a = B ** A

# Print the sum of A^B and B^A
print(a_to_b + b_to_a)