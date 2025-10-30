def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]
    
    q = 0
    for i in range(H):
        for j in range(W):
            if S[i][j] == '?':
                q += 1
    
    count = 0
    
    def check(grid):
        for i in range(H):
            for j in range(W):
                if i > 0 and grid[i][j] == grid[i-1][j]:
                    return False
                if j > 0 and grid[i][j] == grid[i][j-1]:
                    return False
        return True
    
    def generate_grids(index, current_grid):
        nonlocal count
        
        if index == H * W:
            if check(current_grid):
                count = (count + 1) % 998244353
            return
        
        row = index // W
        col = index % W
        
        if S[row][col] != '?':
            generate_grids(index + 1, current_grid)
        else:
            for val in ['1', '2', '3']:
                new_grid = [list(row) for row in current_grid]
                new_grid[row][col] = val
                generate_grids(index + 1, new_grid)
    
    initial_grid = [list(row) for row in S]
    generate_grids(0, initial_grid)
    
    print(count)

solve()