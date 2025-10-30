# YOUR CODE HERE
# Read the input
A = list(map(int, input().split()))

# Calculate the sum A_0 * 2^0 + A_1 * 2^1 + ... + A_63 * 2^63
result = 0
for i in range(64):
    result += A[i] * (2 ** i)

print(result)