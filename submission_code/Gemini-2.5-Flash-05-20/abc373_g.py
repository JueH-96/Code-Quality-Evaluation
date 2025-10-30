import sys

def solve():
    N = int(sys.stdin.readline())

    P_points = []
    for i in range(N):
        A, B = map(int, sys.stdin.readline().split())
        P_points.append((A, B, i)) # Store (x, y, original_index)

    Q_points = []
    for i in range(N):
        C, D = map(int, sys.stdin.readline().split())
        Q_points.append((C, D, i)) # Store (x, y, original_index)

    # Sort P_points by x-coordinate, then y-coordinate (ascending).
    # Python's default tuple sort works lexicographically, which is perfect here.
    P_points.sort()

    # This array will store the original Q index (plus 1 for 1-based output)
    # matched_R[p_original_idx] = q_original_idx + 1
    matched_R = [-1] * N
    
    # Keep track of which Q points have been used.
    # The index here corresponds to the original_index of the Q point.
    used_Q_original_indices = [False] * N

    # Iterate through each P point in its sorted order.
    for p_x, p_y, p_original_idx in P_points:
        best_q_original_idx = -1
        min_q_y = float('inf') # Initialize with a very large value

        # Iterate through all Q points to find the best match for the current P point.
        # The "best" match is the one with the minimum y-coordinate among available Q points.
        for q_x, q_y, q_original_idx in Q_points:
            if not used_Q_original_indices[q_original_idx]:
                # If there's a tie in y-coordinates, any choice is fine based on the theorem.
                # Python's default behavior for finding min (first encountered) is sufficient.
                if q_y < min_q_y:
                    min_q_y = q_y
                    best_q_original_idx = q_original_idx
        
        # Assign the found Q point to the current P point's original index.
        # Store as 1-based index as required by output format.
        matched_R[p_original_idx] = best_q_original_idx + 1
        
        # Mark the chosen Q point as used.
        used_Q_original_indices[best_q_original_idx] = True

    # Print the resulting permutation R, space-separated.
    sys.stdout.write(" ".join(map(str, matched_R)) + "
")

solve()