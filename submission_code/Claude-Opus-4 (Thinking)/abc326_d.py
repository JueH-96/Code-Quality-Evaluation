# YOUR CODE HERE
n = int(input())
r = input().strip()
c = input().strip()

# Check if counts match
from collections import Counter
if Counter(r) != Counter(c):
    print("No")
else:
    # Try to fill the grid
    grid = [['.' for _ in range(n)] for _ in range(n)]
    
    def get_first_in_row(row):
        for j in range(n):
            if grid[row][j] != '.':
                return grid[row][j]
        return None
    
    def get_first_in_col(col):
        for i in range(n):
            if grid[i][col] != '.':
                return grid[i][col]
        return None
    
    def count_in_row(row, char):
        return sum(1 for j in range(n) if grid[row][j] == char)
    
    def count_in_col(col, char):
        return sum(1 for i in range(n) if grid[i][col] == char)
    
    def is_valid():
        # Check row constraints
        for i in range(n):
            first = get_first_in_row(i)
            if first is not None and first != r[i]:
                return False
            for char in ['A', 'B', 'C']:
                if count_in_row(i, char) > 1:
                    return False
        
        # Check column constraints
        for j in range(n):
            first = get_first_in_col(j)
            if first is not None and first != c[j]:
                return False
            for char in ['A', 'B', 'C']:
                if count_in_col(j, char) > 1:
                    return False
        
        return True
    
    def is_complete():
        # Check if all constraints are satisfied
        for i in range(n):
            first = get_first_in_row(i)
            if first != r[i]:
                return False
            for char in ['A', 'B', 'C']:
                if count_in_row(i, char) != 1:
                    return False
        
        for j in range(n):
            first = get_first_in_col(j)
            if first != c[j]:
                return False
            for char in ['A', 'B', 'C']:
                if count_in_col(j, char) != 1:
                    return False
        
        return True
    
    def backtrack(pos):
        if pos == n * n:
            return is_complete()
        
        row = pos // n
        col = pos % n
        
        # Try placing each character or leaving empty
        for char in ['.', 'A', 'B', 'C']:
            grid[row][col] = char
            if is_valid():
                if backtrack(pos + 1):
                    return True
            grid[row][col] = '.'
        
        return False
    
    if backtrack(0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")