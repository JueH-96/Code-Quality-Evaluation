# YOUR CODE HERE
def check_bingo(grid, n):
    # Check rows and columns
    for i in range(n):
        if all(grid[i][j] for j in range(n)) or all(grid[j][i] for j in range(n)):
            return True
    
    # Check diagonals
    if all(grid[i][i] for i in range(n)) or all(grid[i][n-1-i] for i in range(n)):
        return True
    
    return False

def solve_bingo():
    n, t = map(int, input().split())
    numbers = list(map(int, input().split()))
    
    grid = [[False] * n for _ in range(n)]
    
    for turn, num in enumerate(numbers, 1):
        i, j = divmod(num - 1, n)
        grid[i][j] = True
        
        if check_bingo(grid, n):
            return turn
    
    return -1

print(solve_bingo())