def solve():
    N = int(input())
    R = input().strip()
    C = input().strip()
    
    # Check if solution is possible
    from collections import Counter
    r_count = Counter(R)
    c_count = Counter(C)
    
    if r_count != c_count:
        print("No")
        return
    
    # Initialize grid
    grid = [['.' for _ in range(N)] for _ in range(N)]
    
    def backtrack(pos):
        if pos == N * N:
            # Check if solution is valid
            # Each row and column should have exactly one A, B, C
            for r in range(N):
                chars = [grid[r][c] for c in range(N) if grid[r][c] != '.']
                if sorted(chars) != ['A', 'B', 'C']:
                    return False
                # Check leftmost constraint
                leftmost_char = None
                for c in range(N):
                    if grid[r][c] != '.':
                        leftmost_char = grid[r][c]
                        break
                if leftmost_char != R[r]:
                    return False
            
            for c in range(N):
                chars = [grid[r][c] for r in range(N) if grid[r][c] != '.']
                if sorted(chars) != ['A', 'B', 'C']:
                    return False
                # Check topmost constraint
                topmost_char = None
                for r in range(N):
                    if grid[r][c] != '.':
                        topmost_char = grid[r][c]
                        break
                if topmost_char != C[c]:
                    return False
            
            return True
        
        r, c = pos // N, pos % N
        
        # Try empty cell
        if backtrack(pos + 1):
            return True
        
        # Try each character
        for char in ['A', 'B', 'C']:
            # Check if we can place this character
            row_count = sum(1 for cc in range(N) if grid[r][cc] == char)
            col_count = sum(1 for rr in range(N) if grid[rr][c] == char)
            
            if row_count == 0 and col_count == 0:
                grid[r][c] = char
                if backtrack(pos + 1):
                    return True
                grid[r][c] = '.'
        
        return False
    
    if backtrack(0):
        print("Yes")
        for row in grid:
            print(''.join(row))
    else:
        print("No")

solve()