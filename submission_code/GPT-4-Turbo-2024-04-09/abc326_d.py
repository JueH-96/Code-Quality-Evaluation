def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    R = data[1]
    C = data[2]
    
    # Check if the requirements for the first row and first column can be met
    if len(set(R)) != N or len(set(C)) != N:
        print("No")
        return
    
    # Create a grid of N x N filled with '.'
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    # Place the required characters in the first cell of each row and column
    for i in range(N):
        grid[i][0] = R[i]
        grid[0][i] = C[i]
    
    # Check if the first row and first column are compatible
    if any(grid[0][i] != C[i] for i in range(N)):
        print("No")
        return
    
    # We need to fill the grid such that each row and each column contains exactly one A, one B, and one C
    from itertools import permutations
    
    # Generate all permutations of 'ABC' that can be used to fill the rows
    all_perms = list(permutations('ABC'))
    
    # Function to check if a permutation is valid for a specific row
    def is_valid_permutation(row, perm):
        if grid[row][0] != perm[0]:  # The first character must match the required character
            return False
        return True
    
    # Try to assign permutations to rows
    def assign_rows(row, used_perms):
        if row == N:
            # Check columns
            for col in range(N):
                col_chars = {grid[r][col] for r in range(N)}
                if len(col_chars) != 3 or col_chars != {'A', 'B', 'C'}:
                    return False
            return True
        
        for perm in all_perms:
            if perm not in used_perms and is_valid_permutation(row, perm):
                # Place the permutation in the row
                for col in range(N):
                    grid[row][col] = perm[col]
                
                used_perms.add(perm)
                if assign_rows(row + 1, used_perms):
                    return True
                used_perms.remove(perm)
                
                # Reset the row
                for col in range(N):
                    grid[row][col] = '.'
        
        return False
    
    if assign_rows(1, {tuple(grid[0])}):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")