from itertools import permutations

def check_solution(N, R, C, grid):
    # Check rows and columns
    for i in range(N):
        if grid[i][0] != R[i] or grid[0][i] != C[i]:
            return False
        if len(set(grid[i])) != 3 or len(set(grid[j][i] for j in range(N))) != 3:
            return False

    # Check all characters are A, B, C or .
    for row in grid:
        for char in row:
            if char not in "ABC.":
                return False

    return True

def solve(N, R, C):
    # Generate all permutations of "ABC"
    perms = list(permutations("ABC"))
    
    for perm in perms:
        # Create initial grid with leftmost and topmost characters
        grid = [[R[i] if j == 0 else '.' for j in range(N)] for i in range(N)]
        for i in range(N):
            grid[0][i] = C[i]
        
        # Try to fill the grid
        for i in range(1, N):
            for j in range(1, N):
                if grid[i][j] == '.':
                    for p in perm:
                        if p not in grid[i] and p not in [grid[k][j] for k in range(N)]:
                            grid[i][j] = p
                            break
                    else:
                        break
            else:
                continue
            break
        else:
            if check_solution(N, R, C, grid):
                return ["Yes"] + ["".join(row) for row in grid]
    
    return ["No"]

# Read input
N = int(input())
R = input()
C = input()

# Solve and print result
result = solve(N, R, C)
print("
".join(result))