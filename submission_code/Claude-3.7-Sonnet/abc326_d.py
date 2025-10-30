# YOUR CODE HERE
from itertools import combinations, permutations

def solve():
    N = int(input())
    R = input()
    C = input()
    
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    def is_valid_partial_grid(r):
        # Check if each row has exactly one 'A', one 'B', and one 'C'
        for i in range(r):
            a_count = sum(1 for j in range(N) if grid[i][j] == 'A')
            b_count = sum(1 for j in range(N) if grid[i][j] == 'B')
            c_count = sum(1 for j in range(N) if grid[i][j] == 'C')
            if a_count != 1 or b_count != 1 or c_count != 1:
                return False
        
        # Check if each column has at most one 'A', one 'B', and one 'C'
        for j in range(N):
            a_count = sum(1 for i in range(r) if grid[i][j] == 'A')
            b_count = sum(1 for i in range(r) if grid[i][j] == 'B')
            c_count = sum(1 for i in range(r) if grid[i][j] == 'C')
            if a_count > 1 or b_count > 1 or c_count > 1:
                return False
        
        # Check the leftmost character of each filled row
        for i in range(r):
            leftmost = None
            for j in range(N):
                if grid[i][j] in ['A', 'B', 'C']:
                    leftmost = grid[i][j]
                    break
            if leftmost != R[i]:
                return False
        
        # Check the topmost character of each column
        for j in range(N):
            topmost = None
            for i in range(r):
                if grid[i][j] in ['A', 'B', 'C']:
                    topmost = grid[i][j]
                    break
            if topmost is not None and topmost != C[j]:
                return False
        
        return True
    
    def backtrack(r):
        if r == N:
            # Check if each column has exactly one 'A', one 'B', and one 'C'
            for j in range(N):
                a_count = sum(1 for i in range(N) if grid[i][j] == 'A')
                b_count = sum(1 for i in range(N) if grid[i][j] == 'B')
                c_count = sum(1 for i in range(N) if grid[i][j] == 'C')
                if a_count != 1 or b_count != 1 or c_count != 1:
                    return False
            return True
        
        # Try all possible ways to place 'A', 'B', and 'C' in the current row
        for positions in combinations(range(N), 3):
            # For each arrangement of 'A', 'B', and 'C'
            for chars in permutations(['A', 'B', 'C']):
                # Clear the current row
                for j in range(N):
                    grid[r][j] = '.'
                
                # Place 'A', 'B', and 'C' in the chosen positions
                for char, pos in zip(chars, positions):
                    grid[r][pos] = char
                
                # Check if the leftmost character matches R[r]
                leftmost_pos = min(positions)
                leftmost_char = grid[r][leftmost_pos]
                if leftmost_char != R[r]:
                    continue
                
                if is_valid_partial_grid(r+1) and backtrack(r+1):
                    return True
        
        # Clear the current row before backtracking
        for j in range(N):
            grid[r][j] = '.'
        
        return False
    
    if backtrack(0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

solve()