import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
Y = int(data[2])

A = []
index = 3
for i in range(H):
    row = list(map(int, data[index:index+W]))
    A.append(row)
    index += W

def sink_island(sea_level):
    visited = [[False for _ in range(W)] for _ in range(H)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(x, y):
        if x < 0 or x >= H or y < 0 or y >= W or visited[x][y] or A[x][y] <= sea_level:
            return
        visited[x][y] = True
        for dx, dy in directions:
            dfs(x + dx, y + dy)
    
    for i in range(H):
        for j in range(W):
            if not visited[i][j] and A[i][j] <= sea_level:
                dfs(i, j)
    
    count = 0
    for i in range(H):
        for j in range(W):
            if not visited[i][j]:
                count += 1
    return count

results = []
for year in range(1, Y + 1):
    results.append(sink_island(year))

for result in results:
    print(result)