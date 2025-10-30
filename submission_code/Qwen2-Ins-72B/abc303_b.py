N, M = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(M)]

# ある2人が隣に立ったことがあるかのフラグ
flag = [[False] * N for _ in range(N)]
for a in A:
    for i in range(N-1):
        flag[a[i]-1][a[i+1]-1] = True
        flag[a[i+1]-1][a[i]-1] = True

# 一度も隣に立ったことがない2人を見つける
ans = 0
for i in range(N):
    for j in range(i+1, N):
        if not flag[i][j]:
            ans += 1

print(ans)