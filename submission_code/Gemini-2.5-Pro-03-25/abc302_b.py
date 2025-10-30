# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read grid dimensions H and W from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid content into a list of strings. S[i] will store the i-th row.
    S = [sys.stdin.readline().strip() for _ in range(H)]
    
    # The target sequence of characters we are looking for
    target = "snuke"
    
    # The 8 possible directions (horizontal, vertical, diagonal) for the sequence.
    # Each tuple represents the constant displacement (delta_row, delta_col) 
    # between consecutive cells in the sequence.
    # The displacement must satisfy max(|delta_row|, |delta_col|) = 1 and not be (0,0).
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # Up-Left, Up, Up-Right
        (0, -1),           (0, 1),   # Left, Right
        (1, -1), (1, 0), (1, 1)    # Down-Left, Down, Down-Right
    ]
    
    # Iterate through each cell (r, c) in the grid using 0-based indexing.
    # Each cell is considered as a potential starting cell (A_1) of the sequence.
    for r in range(H):
        for c in range(W):
            # Check if the character at the current cell (r, c) matches the 
            # first character of the target sequence ('s').
            if S[r][c] == target[0]:
                # If the character is 's', it could be the start of the sequence.
                # Explore the 8 possible directions from this cell.
                for dr, dc in directions:
                    # List to store the 1-based coordinates of the cells forming the potential sequence.
                    # We use 1-based coordinates for the output as required.
                    path = []
                    # Assume this direction potentially leads to a valid sequence until proven otherwise.
                    match = True
                    
                    # Check the sequence of 5 cells starting from (r, c) in the direction (dr, dc).
                    # The sequence has length 5 (equal to length of 'snuke').
                    # The k-th cell in the sequence (0-indexed, k=0..4) is at 
                    # 0-based coordinates (r + k*dr, c + k*dc).
                    for k in range(len(target)): # len(target) is 5
                        # Calculate the 0-based coordinates of the k-th cell in the sequence.
                        curr_r = r + k * dr
                        curr_c = c + k * dc
                        
                        # Check if the calculated coordinates are within the grid boundaries.
                        # Row index must be in [0, H-1] and column index must be in [0, W-1].
                        if not (0 <= curr_r < H and 0 <= curr_c < W):
                            # If the cell is outside the grid boundaries, this direction is invalid.
                            match = False
                            break # Stop checking further cells in this direction.
                        
                        # Check if the character at the cell (curr_r, curr_c) matches the k-th 
                        # character of the target sequence (target[k]).
                        if S[curr_r][curr_c] != target[k]:
                            # If the character doesn't match, this direction is invalid.
                            match = False
                            break # Stop checking further cells in this direction.
                        
                        # If both boundary check and character check pass, this cell is part 
                        # of the potential sequence. Store its 1-based coordinates.
                        # Convert 0-based (curr_r, curr_c) to 1-based (R, C) by adding 1.
                        path.append((curr_r + 1, curr_c + 1))

                    # After checking all 5 required cells for the current starting cell (r, c)
                    # and current direction (dr, dc):
                    # If the match flag is still True, it means we found a valid sequence 
                    # matching all conditions.
                    if match:
                        # The problem statement guarantees that there is a unique solution.
                        # Therefore, the first valid sequence we find must be the correct one.
                        # Print the 1-based coordinates of the cells in the found sequence, 
                        # one pair per line.
                        for R, C in path:
                            print(f"{R} {C}")
                        # Terminate the function (and the program) since the unique solution is found.
                        return 

# Call the main function `solve()` to start the process.
solve()