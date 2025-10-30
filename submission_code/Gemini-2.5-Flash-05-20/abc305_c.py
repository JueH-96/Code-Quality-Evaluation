import sys

# Function to solve the problem
def solve():
    # Read H and W from the first line of input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid into a list of strings
    S = []
    for _ in range(H):
        S.append(sys.stdin.readline().strip())
        
    # Initialize min/max row and column indices for cookies found.
    # We use H and W as initial maximums for min_r and min_c,
    # and -1 as initial minimums for max_r and max_c.
    # This ensures that any valid 0-indexed row/column (0 to H-1 or 0 to W-1)
    # will correctly update these bounds.
    min_r, max_r = H, -1
    min_c, max_c = W, -1
    
    # First pass: Iterate through the grid to find the bounding box
    # of all '#' characters (existing cookies).
    # This bounding box corresponds to the original rectangle's dimensions.
    for r in range(H):
        for c in range(W):
            if S[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                
    # Second pass: Iterate only within the identified bounding box.
    # The problem guarantees that exactly one cookie was eaten.
    # Therefore, within this bounding box, there will be exactly one '.' character.
    # This '.' character represents the location of the eaten cookie.
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if S[r][c] == '.':
                # Found the empty spot.
                # Convert 0-indexed (r, c) to 1-indexed (r+1, c+1) for output.
                print(r + 1, c + 1)
                # Since the answer is unique, we can exit immediately after finding it.
                return 

# Call the solve function to run the program
solve()