# Read the input as a list of integers
a = list(map(int, input().split()))

# Compute the result by iterating through each bit and its position
result = 0
for i in range(64):
    result += a[i] * (2 ** i)

# Print the result
print(result)