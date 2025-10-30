def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    H, W = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # Directions for 8-connected neighbours.
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def dfs(x, y):
        stack = [(x, y)]
        grid[x][y] = '.'  # Mark as visited.
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                    grid[nx][ny] = '.'
                    stack.append((nx, ny))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                dfs(i, j)
                count += 1
    print(count)
    
if __name__ == '__main__':
    main()