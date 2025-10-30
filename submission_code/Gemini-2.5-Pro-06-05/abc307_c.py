# YOUR CODE HERE
import sys

def solve():
    """
    Main function to read input, solve the problem, and print the output.
    """

    # Helper function to read a sheet's dimensions and pattern from stdin,
    # and return a set of coordinates for the black squares ('#').
    def get_points_from_stdin():
        try:
            line = sys.stdin.readline()
            if not line: return set() # Handle end of input
            H, W = map(int, line.split())
            points = set()
            for r in range(H):
                row_str = sys.stdin.readline().strip()
                for c in range(W):
                    if row_str[c] == '#':
                        points.add((r, c))
            return points
        except (IOError, ValueError):
            return None # Should not happen with valid input format

    # Read the three sheets into sets of coordinates.
    points_A = get_points_from_stdin()
    points_B = get_points_from_stdin()
    points_X = get_points_from_stdin()
    
    # Problem constraints guarantee sheets are non-empty and input is valid.

    # Helper function to find all valid shifts (dr, dc) that would make
    # sheet P a sub-pattern of sheet X.
    def find_valid_shifts(points_P, points_target):
        # A sheet cannot fit into a target with fewer black squares.
        if len(points_P) > len(points_target):
            return set()
        
        # Constraints guarantee at least one black square, so points_P is not empty.
        # Pick an arbitrary point from P to act as an anchor.
        pr_anchor, pc_anchor = next(iter(points_P))
        
        valid_shifts = set()
        
        # Try to map the anchor point to every point in the target.
        # Each mapping defines a potential shift.
        for xr_target, xc_target in points_target:
            dr = xr_target - pr_anchor
            dc = xc_target - pc_anchor
            
            # Verify if this shift works for all other points in P.
            is_valid_shift = True
            for pr, pc in points_P:
                if (pr + dr, pc + dc) not in points_target:
                    is_valid_shift = False
                    break
            
            if is_valid_shift:
                valid_shifts.add((dr, dc))
                
        return valid_shifts

    # Find all possible valid placements for sheets A and B within X.
    valid_shifts_A = find_valid_shifts(points_A, points_X)
    valid_shifts_B = find_valid_shifts(points_B, points_X)

    # Check every combination of valid placements for A and B.
    for dr_A, dc_A in valid_shifts_A:
        for dr_B, dc_B in valid_shifts_B:
            
            # Apply the shifts to A and B to get their coordinates on the final canvas.
            shifted_A = {(r + dr_A, c + dc_A) for r, c in points_A}
            shifted_B = {(r + dr_B, c + dc_B) for r, c in points_B}
            
            # If the union of the two shifted sheets exactly matches the target sheet X,
            # we have found a valid construction.
            if shifted_A.union(shifted_B) == points_X:
                print("Yes")
                return
    
    # If the loops complete without finding a solution, it's not possible.
    print("No")

# Run the solution
solve()