# Read input values
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Extract multiples of K, divide by K, and sort the results
result = sorted([a // K for a in A if a % K == 0])

# Print the result with spaces in between
print(' '.join(map(str, result)))