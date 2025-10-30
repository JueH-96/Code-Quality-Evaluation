import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    target = "snuke"
    target_len = len(target) # This is 5

    # Iterate over all possible starting cells (r_start, c_start)
    # These coordinates are 0-indexed.
    # (r_start, c_start) is the potential location of 's' (target[0]).
    for r_start in range(H):
        for c_start in range(W):
            # Try all 8 non-zero directions (dr, dc)
            # dr is the change in row index for each step.
            # dc is the change in column index for each step.
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        # A step of (0,0) means staying in the same cell.
                        # This is not a valid progression for distinct cells forming "snuke".
                        continue

                    path_coords = [] # Stores 0-indexed (row, col) of cells in the current path
                    current_path_is_valid = True

                    # Try to form the word "snuke" along this direction
                    for k in range(target_len): # k iterates from 0 to 4 (for 5 characters)
                        # Calculate coordinates for the k-th character of "snuke"
                        # Cell for target[k] is (r_k, c_k)
                        # For k=0: (r_k, c_k) = (r_start, c_start)
                        # For k=1: (r_k, c_k) = (r_start + dr, c_start + dc)
                        # ...
                        # For k=4: (r_k, c_k) = (r_start + 4*dr, c_start + 4*dc)
                        r_k = r_start + k * dr
                        c_k = c_start + k * dc

                        # Check 1: Is (r_k, c_k) within the grid boundaries?
                        if not (0 <= r_k < H and 0 <= c_k < W):
                            current_path_is_valid = False
                            break # This path segment goes out of bounds, so try next direction

                        # Check 2: Does the character at grid[r_k][c_k] match target[k]?
                        if grid[r_k][c_k] != target[k]:
                            current_path_is_valid = False
                            break # Character mismatch, so try next direction
                        
                        # If both checks pass, this cell is valid for this path segment.
                        # Add its 0-indexed coordinates to our list.
                        path_coords.append((r_k, c_k))
                    
                    # After checking all 5 characters for the current path:
                    if current_path_is_valid:
                        # A valid "snuke" sequence has been found.
                        # The problem guarantees a unique solution, so this must be it.
                        # Print the 1-indexed coordinates and terminate.
                        for r_coord, c_coord in path_coords:
                            # Convert 0-indexed (r_coord, c_coord) to 1-indexed for output
                            sys.stdout.write(f"{r_coord + 1} {c_coord + 1}
")
                        return # Exit the function, which ends the program

# Call the main solving function
solve()