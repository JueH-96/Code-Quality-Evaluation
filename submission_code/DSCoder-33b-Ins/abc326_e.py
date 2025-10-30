# YOUR CODE HERE

# Read the inputs
N = int(input())
A = list(map(int, input().split()))

# Sort the list in descending order
A.sort(reverse=True)

# Initialize the expected value
expected_value = 0

# Calculate the expected value
for i in range(N):
    expected_value += A[i] * (i+1)

# Print the answer modulo 998244353
print(expected_value % 998244353)