# Read the input as a list of integers
a = list(map(int, input().split()))

# Calculate the sum
result = 0
for i in range(64):
    result += a[i] * (1 << i)  # Using bit shift for efficiency

# Print the result
print(result)