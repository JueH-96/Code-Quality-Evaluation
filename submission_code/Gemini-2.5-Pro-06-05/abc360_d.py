import sys
from bisect import bisect_left

def main():
    """
    Solves the ant crossing problem using a sweep-line algorithm.
    """
    # Use fast I/O
    input = sys.stdin.readline
    
    # Read problem parameters
    try:
        N, T = map(int, input().split())
        S = input().strip()
        X = list(map(int, input().split()))
    except (IOError, ValueError):
        # Gracefully exit on empty or malformed input
        return

    # Combine ant data into a list of (position, direction) tuples.
    # Direction: 1 for positive, 0 for negative.
    ants = []
    for i in range(N):
        ants.append((X[i], int(S[i])))

    # Sort ants by their initial position to enable a sweep-line approach.
    ants.sort()

    total_crossings = 0
    
    # This list will store the positions of right-moving ants encountered so far.
    # Since we process ants sorted by position, this list is inherently sorted.
    right_movers_positions = []
    
    # Pre-calculate 2*T for efficiency
    two_T = 2 * T

    # Sweep from left to right through the sorted ants
    for pos, direction in ants:
        if direction == 1:
            # Encountered a right-moving ant. Store its position.
            right_movers_positions.append(pos)
        else:  # direction == 0
            # Encountered a left-moving ant at position `pos`.
            # A right-moving ant at `r_pos` will cross it if `r_pos >= pos - 2*T`.
            # We count how many previously seen right-movers satisfy this.
            threshold = pos - two_T
            
            # `bisect_left` finds the index of the first element in
            # `right_movers_positions` that is >= `threshold`.
            idx = bisect_left(right_movers_positions, threshold)
            
            # All elements from `idx` to the end of the list satisfy the condition.
            # The number of such elements is the total count of right-movers seen
            # minus the count of those smaller than the threshold (which is `idx`).
            crossings_with_this_ant = len(right_movers_positions) - idx
            
            total_crossings += crossings_with_this_ant

    # Print the final count of crossing pairs.
    print(total_crossings)

if __name__ == "__main__":
    main()