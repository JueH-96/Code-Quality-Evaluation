import sys
input = sys.stdin.read

def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    visited = [[False] * W for _ in range(H)]
    
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '#':
                    visited[nx][ny] = True
                    stack.append((nx, ny))
    
    num_sensors = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                visited[i][j] = True
                dfs(i, j)
                num_sensors += 1
    
    print(num_sensors)

if __name__ == "__main__":
    main()