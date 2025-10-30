# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]

for j in range(M):
    if sum(X[i][j] for i in range(N)) < A[j]:
        print("No")
        break
else:
    print("Yes")