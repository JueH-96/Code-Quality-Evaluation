def solve_grid(N, R, C):
    # Initialize the grid with empty cells
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    # Create sets to track used characters in rows and columns
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    
    # Fill the grid according to R and C
    for i in range(N):
        # Place the character from R in the corresponding row
        grid[i][0] = R[i]
        row_used[i].add(R[i])
        col_used[0].add(R[i])
        
        # Place the character from C in the corresponding column
        grid[0][i] = C[i]
        row_used[0].add(C[i])
        col_used[i].add(C[i])
    
    # Now fill the rest of the grid
    for i in range(N):
        for j in range(N):
            if grid[i][j] == '.':
                # Determine what can be placed in this cell
                for char in 'ABC':
                    if char not in row_used[i] and char not in col_used[j]:
                        grid[i][j] = char
                        row_used[i].add(char)
                        col_used[j].add(char)
                        break
    
    # Check if the grid is valid
    for i in range(N):
        if len(row_used[i]) != 3 or len(col_used[i]) != 3:
            return "No"
    
    # Prepare the output
    result = ["Yes"]
    for row in grid:
        result.append(''.join(row))
    
    return "
".join(result)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
R = data[1]
C = data[2]

# Solve the problem and print the result
print(solve_grid(N, R, C))