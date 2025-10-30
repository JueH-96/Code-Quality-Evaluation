import sys

def find_sensors(h, w, grid):
    def dfs(x, y):
        if x < 0 or x >= h or y < 0 or y >= w or grid[x][y] != '#':
            return
        grid[x] = grid[x][:y] + '.' + grid[x][y+1:]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                dfs(x + dx, y + dy)
    
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                dfs(i, j)
                count += 1
    return count

input = sys.stdin.read
data = input().split()

h, w = int(data[0]), int(data[1])
grid = data[2:2+h]

print(find_sensors(h, w, grid))