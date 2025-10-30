import sys

def solve():
    # Read H, W, N (N_moves to avoid conflict with problem's N for moves)
    H, W, N_moves = map(int, sys.stdin.readline().split())
    # Read movement string T
    T = sys.stdin.readline().strip()
    # Read grid S
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Calculate relative path points
    # path_points[k] will store the (dr, dc) tuple representing the displacement
    # from the starting position after k moves.
    # path_points[0] is (0,0) for the initial position (before any moves).
    path_points = []
    current_r_offset = 0
    current_c_offset = 0
    path_points.append((current_r_offset, current_c_offset)) # Position at step 0 (initial)

    # Populate path_points with displacements for each move
    for move in T:
        if move == 'U':
            current_r_offset -= 1
        elif move == 'D':
            current_r_offset += 1
        elif move == 'L':
            current_c_offset -= 1
        elif move == 'R':
            current_c_offset += 1
        path_points.append((current_r_offset, current_c_offset))

    # Use a set to store distinct final positions
    set_of_final_positions = set()

    # Iterate through all possible cells (r_start, c_start) in the grid
    # These are 0-indexed rows from 0 to H-1, columns from 0 to W-1
    for r_start in range(H):
        for c_start in range(W):
            is_valid_start = True # Assume current (r_start, c_start) is a valid initial position

            # Check all points along the path for validity
            for dr, dc in path_points:
                r_curr = r_start + dr
                c_curr = c_start + dc

                # Rule 1: Check if current position (r_curr, c_curr) is within grid boundaries
                # If it goes out of bounds, it's effectively "sea" and invalidates the path.
                if not (0 <= r_curr < H and 0 <= c_curr < W):
                    is_valid_start = False
                    break # Path is invalid, no need to check further points for this (r_start, c_start)
                
                # Rule 2: Check if current position (r_curr, c_curr) is land ('.')
                # If it's sea ('#'), it invalidates the path.
                if S[r_curr][c_curr] == '#':
                    is_valid_start = False
                    break # Path is invalid, no need to check further points for this (r_start, c_start)
            
            # If all points on the path were valid (land and within bounds)
            if is_valid_start:
                # Calculate the final position for this valid starting point
                # The final position's displacement is the last element in path_points (after N_moves)
                final_dr, final_dc = path_points[N_moves] # Equivalent to path_points[-1]
                r_final = r_start + final_dr
                c_final = c_start + final_dc
                
                # Add the final position to the set of distinct final positions
                set_of_final_positions.add((r_final, c_final))
    
    # Print the total count of distinct final positions
    sys.stdout.write(str(len(set_of_final_positions)) + '
')

solve()