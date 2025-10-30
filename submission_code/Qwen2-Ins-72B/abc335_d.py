N = int(input())
M = N // 2
A = [[0] * N for _ in range(N)]
i, j = M, M
A[i][j] = 'T'
for k in range(1, N*N):
    if k % 2 == 1:
        if i == 0:
            j += 1
        else:
            i -= 1
    else:
        if j == N-1:
            i += 1
        else:
            j += 1
    A[i][j] = k
for i in range(N):
    for j in range(N):
        print(A[i][j], end=' ')
    print()