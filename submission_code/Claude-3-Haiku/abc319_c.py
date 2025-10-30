from itertools import permutations

def check_disappointed(grid):
    # Check rows
    for i in range(3):
        if grid[i][0] == grid[i][1] and grid[i][1] != grid[i][2]:
            return True
    # Check columns
    for j in range(3):
        if grid[0][j] == grid[1][j] and grid[1][j] != grid[2][j]:
            return True
    # Check diagonals
    if grid[0][0] == grid[1][1] and grid[1][1] != grid[2][2]:
        return True
    if grid[2][0] == grid[1][1] and grid[1][1] != grid[0][2]:
        return True
    return False

def solve():
    grid = [list(map(int, input().split())) for _ in range(3)]
    
    # Generate all possible permutations of the grid
    perms = list(permutations(sum(grid, [])))
    
    # Count the number of permutations where Takahashi is not disappointed
    not_disappointed = 0
    for perm in perms:
        perm_grid = [list(perm[i:i+3]) for i in range(0, len(perm), 3)]
        if not check_disappointed(perm_grid):
            not_disappointed += 1
    
    # Calculate the probability
    probability = not_disappointed / len(perms)
    print(f"{probability:.16f}")

solve()