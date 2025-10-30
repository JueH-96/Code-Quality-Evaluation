# Read the input line and convert to a list of integers
bits = list(map(int, input().split()))

# Initialize the result to 0
result = 0

# Iterate over each bit, multiply by 2^i where i is the index
for i in range(64):
    result += bits[i] * (2 ** i)

# Print the computed result
print(result)