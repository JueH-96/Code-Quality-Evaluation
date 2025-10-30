from typing import List

def check_bingo(grid: List[List[bool]]) -> bool:
    # Check rows
    for row in grid:
        if all(cell for cell in row):
            return True
    
    # Check columns
    for j in range(len(grid)):
        if all(grid[i][j] for i in range(len(grid))):
            return True
    
    # Check diagonals
    if all(grid[i][i] for i in range(len(grid))):
        return True
    if all(grid[i][len(grid) - 1 - i] for i in range(len(grid))):
        return True
    
    return False

def solve():
    N, T = map(int, input().split())
    announced = list(map(int, input().split()))
    
    grid = [[False] * N for _ in range(N)]
    
    for turn, num in enumerate(announced, start=1):
        row = (num - 1) // N
        col = (num - 1) % N
        grid[row][col] = True
        
        if check_bingo(grid):
            print(turn)
            return
    
    print(-1)

solve()