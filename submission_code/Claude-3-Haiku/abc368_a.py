# Read the input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Move the bottom K cards to the top
A = A[-K:] + A[:-K]

# Print the result
print(*A)