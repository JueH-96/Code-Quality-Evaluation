import copy

def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]
    
    for i in range(1, n // 2 + 1):
        temp_grid = copy.deepcopy(grid)
        start_index = i - 1
        end_index = n - i
        for r in range(start_index, end_index + 1):
            for c in range(start_index, end_index + 1):
                grid[c][n - 1 - r] = temp_grid[r][c]
                
    for row in grid:
        print("".join(row))

if __name__ == '__main__':
    solve()