N = int(input())
A = [list(map(int, input().split())) for i in range(N)]
res = 1
for i in range(1, N):
    if res >= i:
        res = A[res-1][i-1]
    else:
        res = A[i][res-1]
print(res)