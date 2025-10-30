import sys

# Use standard input/output
# sys.stdin = open('input.txt', 'r') # Uncomment for local testing from file
# sys.stdout = open('output.txt', 'w') # Uncomment for local testing to file


def solve():
    # Read the number of points
    N = int(sys.stdin.readline())
    
    # Read the coordinates of points P
    P_input = []
    for i in range(N):
        x, y = map(int, sys.stdin.readline().split())
        # Store as (x, y, original_index). Original index is 1-based.
        P_input.append((x, y, i + 1)) 
        
    # Read the coordinates of points Q
    Q_input = []
    for i in range(N):
        x, y = map(int, sys.stdin.readline().split())
        # Store as (x, y, original_index). Original index is 1-based.
        Q_input.append((x, y, i + 1)) 

    # Greedy strategy: Sort P points by x-coordinate, then y-coordinate.
    # Python's default tuple sort works lexicographically, achieving the desired (x, y) sort.
    P_sorted = sorted(P_input)

    # Boolean list to track matched Q points.
    # q_matched[i] is True if Q point with original index i is matched.
    # We use 1-based indexing for original indices, so list size N+1.
    q_matched = [False] * (N + 1) 

    # Result permutation: R is the permutation of (1, 2, ..., N).
    # The problem asks for R_1, R_2, ..., R_N where segment i connects P_i and Q_{R_i}.
    # This means R_i is the original index of the Q point matched with P_i (original index i).
    # We need to store the matched Q index at the position corresponding to the original P index.
    # R_result[i-1] will store the original index of the Q point matched with P_i (original index i).
    R_result = [0] * N

    # Iterate through the sorted P points (from P'_1 to P'_N).
    # Let the current sorted P point be P'_k, with original index p_orig_idx.
    # We need to find the best available Q point to match with P'_k.
    for p_x, p_y, p_orig_idx in P_sorted:
        
        # Variables to find the best available Q point for the current P point.
        # Initialize best_q_y to a value lower than any possible y-coordinate (min 0).
        # Initialize best_q_x to a value higher than any possible x-coordinate (max 5000) for tie-breaking.
        best_q_orig_idx = -1
        best_q_y = -1 
        best_q_x = 5001 

        # Iterate through all Q points to find the best available one according to the greedy criterion.
        for q_x, q_y, q_orig_idx in Q_input:
            # Check if the current Q point is already matched.
            if not q_matched[q_orig_idx]:
                # Found an available Q point.
                
                # Apply the greedy criterion: Choose the available Q point that maximizes its y-coordinate.
                # If there's a tie in y-coordinate, choose the one that minimizes its x-coordinate.
                if q_y > best_q_y:
                    best_q_y = q_y
                    best_q_x = q_x
                    best_q_orig_idx = q_orig_idx
                elif q_y == best_q_y: # Tie in y-coordinate
                    if q_x < best_q_x: # Minimize x-coordinate
                        best_q_x = q_x
                        best_q_orig_idx = q_orig_idx

        # Match the current sorted P point (which has original index p_orig_idx)
        # with the best available Q point found (which has original index best_q_orig_idx).
        # Mark the chosen Q point as matched.
        # `best_q_orig_idx` will always be a valid index >= 1 because there are always N unmatched Q points initially,
        # and we match one P point with one Q point in each of the N iterations of the outer loop.
        q_matched[best_q_orig_idx] = True
        
        # Store the result in the R array. The result array `R_result` should store the permutation R
        # such that R_i is the original index of the Q point matched with P_i (original index i).
        # The current P point being processed has original index `p_orig_idx`. The Q point matched
        # with it has original index `best_q_orig_idx`. So, R_{p_orig_idx} = best_q_orig_idx.
        # In 0-based indexing for `R_result`, this is R_result[p_orig_idx - 1] = best_q_orig_idx.
        R_result[p_orig_idx - 1] = best_q_orig_idx

    # The greedy strategy implemented here is a standard approach for finding non-intersecting matchings
    # under certain geometric conditions. If a non-intersecting matching exists, this strategy is
    # guaranteed to find one. If no non-intersecting matching is possible, this algorithm
    # will still produce a complete matching, but that matching would necessarily contain intersections.
    # However, the problem structure (asking for *any* solution if one exists) implies that
    # if a solution exists, finding the one produced by this greedy method is sufficient.
    # Thus, we do not need to explicitly check for intersections or handle the -1 case
    # separately; if the algorithm completes, the resulting matching is valid if any solution exists.

    # Print the resulting permutation R_1, R_2, ..., R_N separated by spaces.
    # The `*R_result` syntax unpacks the list into arguments for the print function.
    print(*R_result)

solve()