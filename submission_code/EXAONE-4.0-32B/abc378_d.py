import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W, K = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    total = 0

    def dfs(i, j, steps, visited):
        if steps == K:
            return 1
        count = 0
        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                if not visited[ni][nj]:
                    visited[ni][nj] = True
                    count += dfs(ni, nj, steps+1, visited)
                    visited[ni][nj] = False
        return count

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                vis = [[False] * W for _ in range(H)]
                vis[i][j] = True
                total += dfs(i, j, 0, vis)
                
    print(total)

if __name__ == "__main__":
    main()