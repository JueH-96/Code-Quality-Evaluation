import sys

# Read the input line from standard input
line = sys.stdin.readline().split()

# Convert the two parts to integers
a = int(line[0])
b = int(line[1])

# Calculate A^B and B^A
a_pow_b = a ** b
b_pow_a = b ** a

# Calculate the sum
result = a_pow_b + b_pow_a

# Print the result
print(result)