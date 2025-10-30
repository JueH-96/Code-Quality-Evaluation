# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Take K cards from bottom and place on top
# The last K elements become the first K elements
result = A[N-K:] + A[:N-K]

print(' '.join(map(str, result)))