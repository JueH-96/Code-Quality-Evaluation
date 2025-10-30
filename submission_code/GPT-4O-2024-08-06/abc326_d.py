# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    R = data[1]
    C = data[2]
    
    # Initialize the grid
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    # Helper function to check if a grid is valid
    def is_valid():
        for i in range(N):
            row_set = set(grid[i])
            col_set = set(grid[j][i] for j in range(N))
            if len(row_set) != 3 or len(col_set) != 3:
                return False
            if 'A' not in row_set or 'B' not in row_set or 'C' not in row_set:
                return False
            if 'A' not in col_set or 'B' not in col_set or 'C' not in col_set:
                return False
        return True
    
    # Backtracking function to fill the grid
    def backtrack(row):
        if row == N:
            if is_valid():
                print("Yes")
                for line in grid:
                    print(''.join(line))
                return True
            return False
        
        # Try placing the leftmost character in the row
        left_char = R[row]
        for col in range(N):
            if grid[row][col] == '.':
                grid[row][col] = left_char
                # Check if this placement is valid for the column
                if grid[0][col] == '.' or grid[0][col] == C[col]:
                    # Try to fill the rest of the row
                    for perm in permutations('ABC'):
                        if perm[0] == left_char:
                            continue
                        for i in range(1, N):
                            grid[row][(col + i) % N] = perm[i - 1]
                        if backtrack(row + 1):
                            return True
                        # Reset the row
                        for i in range(1, N):
                            grid[row][(col + i) % N] = '.'
                # Reset the leftmost character
                grid[row][col] = '.'
        return False
    
    from itertools import permutations
    
    if not backtrack(0):
        print("No")