N, M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]

for i in range(M):
    if sum(X[j][i] for j in range(N)) < A[i]:
        print('No')
        break
else:
    print('Yes')