# Read the input values
A, B = map(int, input().split())

# Calculate A^B and B^A
a_pow_b = A ** B
b_pow_a = B ** A

# Sum the results and print
print(a_pow_b + b_pow_a)