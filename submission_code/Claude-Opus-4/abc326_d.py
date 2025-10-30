def solve():
    N = int(input())
    R = input().strip()
    C = input().strip()
    
    # Initialize grid with empty cells
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    # Try to place characters using backtracking
    def is_valid_placement(row, col, char):
        # Check if this character already exists in the row
        for c in range(N):
            if grid[row][c] == char:
                return False
        
        # Check if this character already exists in the column
        for r in range(N):
            if grid[r][col] == char:
                return False
        
        # Check leftmost constraint for the row
        for c in range(col):
            if grid[row][c] != '.':
                # There's already a character to the left
                return True
        # This would be the leftmost character
        if char != R[row]:
            return False
        
        # Check topmost constraint for the column
        for r in range(row):
            if grid[r][col] != '.':
                # There's already a character above
                return True
        # This would be the topmost character
        if char != C[col]:
            return False
        
        return True
    
    def backtrack(positions):
        if positions == N * 3:  # All characters placed
            return True
        
        # Try to place the next character
        for row in range(N):
            for col in range(N):
                if grid[row][col] == '.':
                    for char in ['A', 'B', 'C']:
                        if is_valid_placement(row, col, char):
                            grid[row][col] = char
                            if backtrack(positions + 1):
                                return True
                            grid[row][col] = '.'
        
        return False
    
    # Alternative approach: Try systematic placement
    # First, place the required leftmost and topmost characters
    
    # Check if solution exists
    def find_solution():
        # Count requirements
        r_count = {'A': R.count('A'), 'B': R.count('B'), 'C': R.count('C')}
        c_count = {'A': C.count('A'), 'B': C.count('B'), 'C': C.count('C')}
        
        # Each character must appear exactly N times total
        for char in ['A', 'B', 'C']:
            if r_count[char] + c_count[char] > N:
                return False
        
        # Try to construct a valid grid
        used_row = [set() for _ in range(N)]
        used_col = [set() for _ in range(N)]
        
        # Place characters systematically
        for i in range(N):
            for j in range(N):
                placed = False
                for char in ['A', 'B', 'C']:
                    if char in used_row[i] or char in used_col[j]:
                        continue
                    
                    # Check if this placement satisfies constraints
                    valid = True
                    
                    # Check leftmost constraint
                    is_leftmost = all(grid[i][k] == '.' for k in range(j))
                    if is_leftmost and R[i] != char:
                        valid = False
                    elif not is_leftmost and j == N-1:
                        # This is the last position in row, check if we've placed R[i]
                        if R[i] not in used_row[i]:
                            valid = False
                    
                    # Check topmost constraint
                    is_topmost = all(grid[k][j] == '.' for k in range(i))
                    if is_topmost and C[j] != char:
                        valid = False
                    elif not is_topmost and i == N-1:
                        # This is the last position in column, check if we've placed C[j]
                        if C[j] not in used_col[j]:
                            valid = False
                    
                    if valid:
                        grid[i][j] = char
                        used_row[i].add(char)
                        used_col[j].add(char)
                        placed = True
                        break
                
                if not placed:
                    # Try backtracking approach
                    return backtrack(0)
        
        # Verify the solution
        for i in range(N):
            if len(used_row[i]) != 3 or len(used_col[i]) != 3:
                return backtrack(0)
        
        return True
    
    if find_solution() or backtrack(0):
        # Verify constraints are met
        valid = True
        
        # Check each row and column has exactly one of each
        for i in range(N):
            row_chars = [grid[i][j] for j in range(N) if grid[i][j] != '.']
            col_chars = [grid[j][i] for j in range(N) if grid[j][i] != '.']
            
            if sorted(row_chars) != ['A', 'B', 'C'] or sorted(col_chars) != ['A', 'B', 'C']:
                valid = False
                break
        
        # Check leftmost characters
        for i in range(N):
            for j in range(N):
                if grid[i][j] != '.':
                    if grid[i][j] != R[i]:
                        valid = False
                    break
        
        # Check topmost characters
        for j in range(N):
            for i in range(N):
                if grid[i][j] != '.':
                    if grid[i][j] != C[j]:
                        valid = False
                    break
        
        if valid:
            print("Yes")
            for row in grid:
                print(''.join(row))
        else:
            print("No")
    else:
        print("No")

solve()