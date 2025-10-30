def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]
    
    for i_op in range(1, n // 2 + 1):
        next_grid = [['' for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                next_grid[r][c] = grid[r][c]
                
        for r in range(i_op, n + 1 - i_op + 1):
            for c in range(i_op, n + 1 - i_op + 1):
                next_grid[r-1][c-1] = grid[n - c][r-1]
                
        grid = next_grid
        
    for r in range(n):
        print("".join(grid[r]))

if __name__ == '__main__':
    solve()