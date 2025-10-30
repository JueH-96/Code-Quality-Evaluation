# YOUR CODE HERE
import sys

def solve(N, R, C):
    # Initialize the grid with empty cells
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    # Try to place the characters according to the constraints
    for i in range(N):
        grid[i][0] = R[i]
        grid[0][i] = C[i]
    
    # Check if the first row and first column are valid
    if len(set(grid[0])) != N or len(set([grid[i][0] for i in range(N)])) != N:
        return "No"
    
    # Try to fill the rest of the grid
    for i in range(N):
        for j in range(N):
            if grid[i][j] == '.':
                for char in 'ABC':
                    if char not in grid[i] and char not in [grid[x][j] for x in range(N)]:
                        grid[i][j] = char
                        break
                else:
                    return "No"
    
    # Verify the solution
    for i in range(N):
        if len(set(grid[i])) != N or len(set([grid[x][i] for x in range(N)])) != N:
            return "No"
    
    # If all checks pass, return the grid
    result = ["Yes"] + [''.join(row) for row in grid]
    return "
".join(result)

# Read input
input = sys.stdin.read().strip().split()
N = int(input[0])
R = input[1]
C = input[2]

# Solve and print the result
print(solve(N, R, C))