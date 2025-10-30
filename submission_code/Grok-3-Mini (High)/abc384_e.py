import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
H = int(data[index])
W = int(data[index+1])
X = int(data[index+2])
index += 3
P = int(data[index]) - 1  # 0-based
Q = int(data[index+1]) - 1  # 0-based
index += 2
S = []
for i in range(H):
    row = list(map(int, data[index:index+W]))
    S.append(row)
    index += W

dx = [-1,-1,-1,0,0,1,1,1]
dy = [-1,0,1,-1,1,-1,0,1]

visited = [[False for _ in range(W)] for _ in range(H)]
visited[P][Q] = True

def dfs(i, j, current_sum):
    max_sum = current_sum
    for d in range(8):
        ni = i + dx[d]
        nj = j + dy[d]
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and current_sum >= X * S[ni][nj] + 1:
            visited[ni][nj] = True
            new_sum = current_sum + S[ni][nj]
            max_sum = max(max_sum, dfs(ni, nj, new_sum))
            visited[ni][nj] = False  # backtrack
    return max_sum

ans = dfs(P, Q, S[P][Q])
print(ans)