# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))

for i in range(1, N+1):
    print(max(0, A[i-1] - i + 1))