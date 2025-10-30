# Read the input line and split into individual bits
bits = list(map(int, input().split()))

# Initialize the result to 0
result = 0

# Iterate over each bit, calculate its contribution and sum them up
for i in range(64):
    result += bits[i] * (2 ** i)

# Print the final result
print(result)