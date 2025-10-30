import sys

def solve():
    # Read dimensions and grid for sheet A
    HA, WA = map(int, sys.stdin.readline().split())
    A_grid = [sys.stdin.readline().strip() for _ in range(HA)]

    # Read dimensions and grid for sheet B
    HB, WB = map(int, sys.stdin.readline().split())
    B_grid = [sys.stdin.readline().strip() for _ in range(HB)]

    # Read dimensions and grid for sheet X
    HX, WX = map(int, sys.stdin.readline().split())
    X_grid = [sys.stdin.readline().strip() for _ in range(HX)]

    def get_black_coords_and_bounds(H, W, grid):
        """
        Extracts coordinates of '#' squares and their min/max row/column indices.
        Assumes grid contains at least one '#' square.
        """
        coords = []
        min_r, max_r = H, -1
        min_c, max_c = W, -1
        
        for r in range(H):
            for c in range(W):
                if grid[r][c] == '#':
                    coords.append((r, c))
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        return coords, min_r, max_r, min_c, max_c

    # Get black square data for A and B
    A_coords, A_min_r, A_max_r, A_min_c, A_max_c = get_black_coords_and_bounds(HA, WA, A_grid)
    B_coords, B_min_r, B_max_r, B_min_c, B_max_c = get_black_coords_and_bounds(HB, WB, B_grid)

    # Iterate through all possible top-left positions for sheet A relative to sheet X's top-left (0,0).
    # (ar, ac) is the coordinate where A's (0,0) is placed.
    # The condition "cut-out sheet includes all black squares of sheets A and B" means that
    # if A[r_a][c_a] is '#', then its position (ar + r_a, ac + c_a) must be within [0, HX-1] x [0, WX-1].
    # This implies:
    #   0 <= ar + r_a  => ar >= -r_a. For all r_a, so ar >= -min_r_a.
    #   ar + r_a < HX  => ar < HX - r_a. For all r_a, so ar <= HX - 1 - max_r_a.
    # So, the range for ar is from -A_min_r to (HX - A_max_r - 1) inclusive.
    # Python's range(start, end) goes up to (end - 1).
    for ar in range(-A_min_r, HX - A_max_r):
        for ac in range(-A_min_c, WX - A_max_c):
            # Iterate through all possible top-left positions for sheet B relative to sheet X's top-left (0,0).
            # Similar logic applies for br and bc.
            for br in range(-B_min_r, HX - B_max_r):
                for bc in range(-B_min_c, WX - B_max_c):
                    
                    # Create a composite sheet (representing the H_X x W_X cut-out area)
                    # and initialize it as entirely transparent.
                    current_composite = [['.' for _ in range(WX)] for _ in range(HX)]

                    # Paste sheet A's black squares onto the composite sheet.
                    for r_a, c_a in A_coords:
                        nr_a, nc_a = ar + r_a, ac + c_a
                        # (nr_a, nc_a) are guaranteed to be within the X bounds [0, HX-1] x [0, WX-1]
                        # due to the careful selection of ranges for ar and ac.
                        current_composite[nr_a][nc_a] = '#'
                    
                    # Paste sheet B's black squares onto the composite sheet.
                    for r_b, c_b in B_coords:
                        nr_b, nc_b = br + r_b, bc + c_b
                        # Similarly, (nr_b, nc_b) are guaranteed to be within X bounds.
                        current_composite[nr_b][nc_b] = '#'

                    # Check if the generated composite sheet matches the target sheet X.
                    matches_X = True
                    for r in range(HX):
                        for c in range(WX):
                            if current_composite[r][c] != X_grid[r][c]:
                                matches_X = False
                                break
                        if not matches_X:
                            break
                    
                    # If a match is found, print "Yes" and terminate the program.
                    if matches_X:
                        sys.stdout.write("Yes
")
                        return

    # If no combination of placements results in a match after checking all possibilities, print "No".
    sys.stdout.write("No
")

# Call the solver function.
solve()