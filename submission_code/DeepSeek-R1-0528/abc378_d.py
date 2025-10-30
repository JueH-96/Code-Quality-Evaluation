import sys
sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return

    first_line = data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    K_val = int(first_line[2])
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    if K_val == 0:
        count_dots = 0
        for i in range(H):
            for j in range(W):
                if grid[i][j] == '.':
                    count_dots += 1
        print(count_dots)
        return
        
    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    starts = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                starts.append((i, j))
                
    def dfs(i, j, steps, visited, H, W, grid, dirs, K_val):
        if steps == K_val:
            return 1
        total = 0
        for di, dj in dirs:
            ni = i + di
            nj = j + dj
            if ni >= 0 and ni < H and nj >=0 and nj < W:
                if grid[ni][nj] == '.' and not visited[ni][nj]:
                    visited[ni][nj] = True
                    total += dfs(ni, nj, steps+1, visited, H, W, grid, dirs, K_val)
                    visited[ni][nj] = False
        return total

    total_ans = 0
    for (i, j) in starts:
        visited = [[False] * W for _ in range(H)]
        visited[i][j] = True
        total_ans += dfs(i, j, 0, visited, H, W, grid, dirs, K_val)
        
    print(total_ans)

if __name__ == '__main__':
    main()