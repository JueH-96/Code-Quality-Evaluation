N = int(input())
A = [list(map(int, input().split())) for _ in range(N)]

ans = 1
for i in range(N):
    ans = A[max(ans, i)][i]
print(ans)