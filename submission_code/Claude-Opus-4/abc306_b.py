# YOUR CODE HERE
# Read the input
A = list(map(int, input().split()))

# Calculate the sum
result = 0
for i in range(64):
    result += A[i] * (2 ** i)

# Print the result
print(result)