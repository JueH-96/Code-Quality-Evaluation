def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    
    grid = []
    for _ in range(H):
        grid.append(data[idx].strip())
        idx += 1
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(i, j, steps, visited):
        if steps == K:
            return 1
        total = 0
        for dx, dy in directions:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and not visited[ni][nj]:
                    visited[ni][nj] = True
                    total += dfs(ni, nj, steps + 1, visited)
                    visited[ni][nj] = False
        return total
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = [[False] * W for _ in range(H)]
                visited[i][j] = True
                count += dfs(i, j, 0, visited)
    
    print(count)

if __name__ == "__main__":
    main()