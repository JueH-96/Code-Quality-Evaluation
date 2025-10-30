# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())

    cells_data = []
    for _ in range(M):
        X, Y, C = sys.stdin.readline().split()
        cells_data.append((int(X), int(Y), C))

    # Initialize bounds for h(r) for each row r.
    # lower_bound[r] stores the minimum possible value for h(r). Default 0.
    # upper_bound[r] stores the maximum possible value for h(r). Default N.
    lower_bound = defaultdict(lambda: 0)
    upper_bound = defaultdict(lambda: N)

    # 1. Process M given cells to set initial bounds
    for X, Y, C in cells_data:
        if C == 'B':
            # If (X, Y) is black, then h(X) must be at least Y.
            lower_bound[X] = max(lower_bound[X], Y)
        else: # C == 'W'
            # If (X, Y) is white, then h(X) must be at most Y-1.
            upper_bound[X] = min(upper_bound[X], Y - 1)

    # Collect all relevant row indices.
    # These include all X_i from input, and the boundary rows 1 and N.
    unique_x = set()
    for X, _, _ in cells_data:
        unique_x.add(X)
    unique_x.add(1) # Row 1 is always relevant for propagation
    unique_x.add(N) # Row N is always relevant for propagation
    
    # Sort them to process rows in order for propagation.
    relevant_rows = sorted(list(unique_x))

    # 2. Propagate lower bounds (bottom-up pass)
    # The condition h(r) >= h(r+1) implies that lower_bound[r] must be at least lower_bound[r+1].
    # Iterate from the second-to-last relevant row down to the first.
    for i in range(len(relevant_rows) - 2, -1, -1):
        r_curr = relevant_rows[i]
        r_next = relevant_rows[i+1]
        
        # The true h(r_curr) must be >= h(r_next).
        # Since h(r_next) >= lower_bound[r_next], we must have h(r_curr) >= lower_bound[r_next].
        # Update lower_bound[r_curr] to reflect this.
        lower_bound[r_curr] = max(lower_bound[r_curr], lower_bound[r_next])

    # 3. Propagate upper bounds (top-down pass)
    # The condition h(r) >= h(r+1) implies that upper_bound[r+1] must be at most upper_bound[r].
    # Iterate from the second relevant row up to the last.
    for i in range(1, len(relevant_rows)):
        r_curr = relevant_rows[i]
        r_prev = relevant_rows[i-1]
        
        # The true h(r_curr) must be <= h(r_prev).
        # Since h(r_prev) <= upper_bound[r_prev], we must have h(r_curr) <= upper_bound[r_prev].
        # Update upper_bound[r_curr] to reflect this.
        upper_bound[r_curr] = min(upper_bound[r_curr], upper_bound[r_prev])

    # 4. Final consistency check
    # After propagation, for every relevant row r, the lower bound must not exceed the upper bound.
    for r in relevant_rows:
        if lower_bound[r] > upper_bound[r]:
            print("No")
            return
            
    # If all checks pass, it means a valid sequence h(r) can be constructed.
    # For example, we could choose h(r) = lower_bound[r] for all r.
    # The non-increasing property of lower_bound is ensured by the bottom-up pass.
    # The condition lower_bound[r] <= upper_bound[r] is checked above.
    print("Yes")

solve()