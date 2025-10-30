# Read the input as a list of integers
bits = list(map(int, input().split()))

# Calculate the sum of each bit multiplied by 2^i
result = 0
for i in range(64):
    result += bits[i] * (2 ** i)

# Output the result
print(result)