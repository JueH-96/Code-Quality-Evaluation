# YOUR CODE HERE

N = int(input())
A = list(map(int, [input() for _ in range(N)]))

print(A[N-1], A[N-2], A[N-3], A[0], sep='
')