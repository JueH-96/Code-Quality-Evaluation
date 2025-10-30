# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by iterating through all possible final positions and
    verifying the validity of the path leading to each.
    """
    
    # Read input: grid dimensions H, W, number of moves N, move sequence T, and the grid S.
    try:
        H, W, N = map(int, sys.stdin.readline().split())
        T = sys.stdin.readline().strip()
        S = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # This part is for local testing and will not be executed in a standard judge environment.
        # It allows running the script with sample inputs without stdin redirection.
        return

    # Counter for the number of possible final positions.
    ans = 0
    
    # A dictionary to map each move to the coordinate change of its *inverse* move.
    # The format is (row_change, column_change).
    # For example, the inverse of 'U' (Up, row-1) is 'D' (Down, row+1), so its delta is (1, 0).
    inv_delta = {
        'L': (0, 1),   # Inverse of Left is Right
        'R': (0, -1),  # Inverse of Right is Left
        'U': (1, 0),   # Inverse of Up is Down
        'D': (-1, 0)   # Inverse of Down is Up
    }

    # Iterate over all interior cells of the grid. The problem guarantees the perimeter is sea,
    # so we only need to consider non-perimeter cells as potential path locations.
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            
            # A valid final position must be a land cell ('.').
            if S[r][c] == '#':
                continue

            # Assume this land cell (r, c) is a valid final position.
            # We now trace the path backward to verify this assumption.
            is_path_valid = True
            
            # Start the backward trace from the potential final position (r, c).
            curr_r, curr_c = r, c
            
            # Iterate through the moves in T in reverse order to trace the path backward.
            for move in reversed(T):
                # Get the coordinate change for the inverse of the current move.
                dr, dc = inv_delta[move]
                
                # Apply the inverse move to find the previous cell in the path.
                curr_r += dr
                curr_c += dc
                
                # If any cell on the backward-traced path is sea ('#'), the path is invalid.
                # The perimeter-is-sea constraint saves us from explicit out-of-bounds checks.
                if S[curr_r][curr_c] == '#':
                    is_path_valid = False
                    break
            
            # If the entire path was on land, this is a valid scenario.
            if is_path_valid:
                ans += 1
    
    # Print the total count of valid final positions.
    print(ans)

# Execute the solution
solve()