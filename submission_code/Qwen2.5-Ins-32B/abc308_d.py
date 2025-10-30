import sys

def solve(h, w, grid):
    snuke = "snuke"
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    
    def dfs(x, y, idx):
        if x == h-1 and y == w-1:
            return True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == snuke[(idx + 1) % 5]:
                if dfs(nx, ny, (idx + 1) % 5):
                    return True
        return False
    
    if grid[0][0] == 's':
        if dfs(0, 0, 0):
            print("Yes")
            return
    print("No")

input = sys.stdin.read
data = input().split()
h, w = int(data[0]), int(data[1])
grid = data[2:2+h]

solve(h, w, grid)