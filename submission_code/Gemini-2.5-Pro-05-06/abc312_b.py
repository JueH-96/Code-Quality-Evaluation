import sys

def is_tak_code(R_start, C_start, grid):
    # Condition 2: Top-left 3x3 area must be black
    # Cells (R_start+i, C_start+j) for i in [0,1,2], j in [0,1,2]
    for i_offset in range(3):
        for j_offset in range(3):
            if grid[R_start + i_offset][C_start + j_offset] != '#':
                return False

    # Condition 2: Bottom-right 3x3 area must be black
    # Cells (R_start+i, C_start+j) for i in [6,7,8], j in [6,7,8]
    for i_offset in range(6, 9):
        for j_offset in range(6, 9):
            if grid[R_start + i_offset][C_start + j_offset] != '#':
                return False

    # Condition 3: White border around top-left 3x3 area
    # Cells grid[R_start+i][C_start+3] for i in [0,1,2] must be '.'
    for i_offset in range(3):
        if grid[R_start + i_offset][C_start + 3] != '.':
            return False
    
    # Cells grid[R_start+3][C_start+j] for j in [0,1,2] must be '.'
    for j_offset in range(3):
        if grid[R_start + 3][C_start + j_offset] != '.':
            return False

    # Cell grid[R_start+3][C_start+3] must be '.'
    if grid[R_start + 3][C_start + 3] != '.':
        return False

    # Condition 3: White border around bottom-right 3x3 area
    # Cells grid[R_start+i][C_start+5] for i in [6,7,8] must be '.'
    for i_offset in range(6, 9):
        if grid[R_start + i_offset][C_start + 5] != '.':
            return False

    # Cells grid[R_start+5][C_start+j] for j in [6,7,8] must be '.'
    for j_offset in range(6, 9):
        if grid[R_start + 5][C_start + j_offset] != '.':
            return False
    
    # Cell grid[R_start+5][C_start+5] must be '.'
    if grid[R_start + 5][C_start + 5] != '.':
        return False
            
    return True

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S_grid = [sys.stdin.readline().strip() for _ in range(N)]

    found_tak_codes = []

    # Iterate R_idx from 0 to N-9 (inclusive)
    # Iterate C_idx from 0 to M-9 (inclusive)
    # N-9+1 is the number of possible start rows/cols if 0-indexed
    for R_idx in range(N - 9 + 1): 
        for C_idx in range(M - 9 + 1): 
            if is_tak_code(R_idx, C_idx, S_grid):
                # Output is 1-indexed
                found_tak_codes.append((R_idx + 1, C_idx + 1))

    for r_ans, c_ans in found_tak_codes:
        print(r_ans, c_ans)

if __name__ == '__main__':
    solve()