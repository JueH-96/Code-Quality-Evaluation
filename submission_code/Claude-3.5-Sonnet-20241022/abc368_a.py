# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Take last K elements and put them at start
result = A[N-K:] + A[:N-K]

print(*result)