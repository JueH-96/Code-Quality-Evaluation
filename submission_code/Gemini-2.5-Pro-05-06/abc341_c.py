import sys

def solve():
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    S_grid = [sys.stdin.readline().strip() for _ in range(H)]

    active_cells = set()

    # Initialize active_cells with all land cells
    # Coordinates are 0-indexed: r from 0 to H-1, c from 0 to W-1
    for r in range(H):
        for c in range(W):
            if S_grid[r][c] == '.':
                active_cells.add((r, c))
    
    move_deltas = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }

    for move_char in T:
        # If at any point no cells are possible, future steps will also yield no cells.
        # This check is an optimization. The problem guarantees final answer >= 1,
        # so active_cells won't be empty *at the end*.
        if not active_cells:
            break 
            
        dr, dc = move_deltas[move_char]
        
        next_active_cells = set()
        
        for r_prev, c_prev in active_cells:
            r_curr = r_prev + dr
            c_curr = c_prev + dc

            # As reasoned in the thought process:
            # If (r_prev, c_prev) is a land cell, it's not on the perimeter.
            # So 1 <= r_prev <= H-2 and 1 <= c_prev <= W-2.
            # This implies r_curr and c_curr will be valid indices within [0, H-1] and [0, W-1].
            # Thus, S_grid[r_curr][c_curr] access is safe without explicit bounds check.
            
            if S_grid[r_curr][c_curr] == '.':
                next_active_cells.add((r_curr, c_curr))
        
        active_cells = next_active_cells
            
    print(len(active_cells))

solve()