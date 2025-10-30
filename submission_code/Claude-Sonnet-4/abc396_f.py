N, M = map(int, input().split())
A = list(map(int, input().split()))

for k in range(M):
    # Count inversions directly
    inversions = 0
    for i in range(N):
        for j in range(i + 1, N):
            bi = (A[i] + k) % M
            bj = (A[j] + k) % M
            if bi > bj:
                inversions += 1
    
    print(inversions)