N, M = map(int, input().split())
A = list(map(int, input().split()))

# For each day i
for i in range(1, N+1):
    # Find first firework day >= i
    for j in range(M):
        if A[j] >= i:
            print(A[j] - i)
            break