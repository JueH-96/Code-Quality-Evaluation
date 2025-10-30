# YOUR CODE HERE
# Read two integers A and B from standard input
A, B = map(int, input().split())

# Calculate A to the power of B
a_pow_b = A ** B

# Calculate B to the power of A
b_pow_a = B ** A

# Calculate the sum
result = a_pow_b + b_pow_a

# Print the final result to standard output
print(result)