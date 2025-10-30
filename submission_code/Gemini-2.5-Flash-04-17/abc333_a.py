import sys

# Read the integer N from standard input
N = int(sys.stdin.readline())

# Convert the integer N to its string representation
digit_str = str(N)

# Concatenate N copies of the digit string
result = digit_str * N

# Print the resulting string to standard output
print(result)