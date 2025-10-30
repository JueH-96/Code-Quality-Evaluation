# YOUR CODE HERE

# Read the input from stdin
A = list(map(int, input().split()))

# Calculate the sum
sum = 0
for i in range(64):
    sum += A[i] * 2**i

# Print the answer
print(sum)