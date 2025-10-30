# YOUR CODE HERE
n, m = map(int, input().split())
grid = [input() for _ in range(n)]
visited = [[False] * m for _ in range(n)]
count = 0

def dfs(r, c):
    global count
    if not (0 <= r < n and 0 <= c < m) or grid[r][c] == '#' or visited[r][c]:
        return
    visited[r][c] = True
    count += 1
    dfs(r + 1, c)
    dfs(r - 1, c)
    dfs(r, c + 1)
    dfs(r, c - 1)

dfs(1, 1)
print(count)