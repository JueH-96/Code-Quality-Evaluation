# Read input from stdin
N = int(input())

# Calculate the number of digits in N
D = len(str(N))

# Calculate the number of digits to truncate
K = max(0, D - 3)

# Compute the result by truncating the last K digits (set them to zero)
result = (N // (10 ** K)) * (10 ** K)

# Print the result to stdout
print(result)