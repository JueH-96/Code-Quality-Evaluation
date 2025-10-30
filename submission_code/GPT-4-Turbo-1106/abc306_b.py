# Read the input sequence of 0s and 1s
A = list(map(int, input().split()))

# Calculate the sum of the sequence with the given formula
result = sum(a * (2 ** i) for i, a in enumerate(A))

# Print the result
print(result)