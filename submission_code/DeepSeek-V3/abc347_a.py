# YOUR CODE HERE
# Read input values
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Extract elements that are multiples of K and divide them by K
result = [a // K for a in A if a % K == 0]

# Sort the result in ascending order
result.sort()

# Print the result with spaces in between
print(' '.join(map(str, result)))