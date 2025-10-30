def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Read P (A_i, B_i)
    P = []
    idx = 1
    for i in range(N):
        x, y = int(input_data[idx]), int(input_data[idx+1])
        idx += 2
        P.append((x, y))
    
    # Read Q (C_i, D_i)
    Q = []
    for i in range(N):
        x, y = int(input_data[idx]), int(input_data[idx+1])
        idx += 2
        Q.append((x, y))

    # We will place all 2N points around a chosen reference O, sort them by angle,
    # and then try to find a non-crossing matching in circular order that pairs each P with a Q.
    # A known approach is to interpret P as '(' and Q as ')', sort them in a circle, 
    # and then look for a rotation that yields a balanced parentheses sequence.
    # Such a balanced matching on the circle is guaranteed to be non-crossing in the plane.
    #
    # If no rotation yields a balanced sequence, we print -1.

    # Pick a reference point O that is safely outside the bounding box (since all coords >= 0 <= 5000)
    # For simplicity, use O = (-1, -1).
    O = (-1, -1)
    
    def angle_and_dist(pt):
        # Returns (angle, dist) from O
        dx = pt[0] - O[0]
        dy = pt[1] - O[1]
        return (math.atan2(dy, dx), dx*dx + dy*dy)

    # Build a single list of (angle, dist, type, original_index)
    # type = '(' for P, ')' for Q
    # original_index is 1-based index of P or Q we will need later
    L = []
    for i, (x, y) in enumerate(P):
        a, d = angle_and_dist((x, y))
        L.append((a, d, '(', i+1))  # i+1 is the P index
    for i, (x, y) in enumerate(Q):
        a, d = angle_and_dist((x, y))
        L.append((a, d, ')', i+1))  # i+1 is the Q index

    # Sort by (angle, dist)
    L.sort(key=lambda x: (x[0], x[1]))

    # Create the parentheses string in sorted order (circular)
    # Also keep track of each item's (type, original_index)
    S = [tp for (_, _, tp, _) in L]

    # To recover the matching, we will need:
    #   - For each position i in 0..2N-1, store the (type, originalIndex).
    # Then when we do the stack-based matching, each pair (openPos, closePos) 
    # corresponds to P = L[openPos], Q = L[closePos].
    # So we have a length N array R such that R[pIndex] = qIndex (both 1-based).
    # pIndex is for the P we matched, qIndex is for the Q we matched.
    data_for_reconstruction = [(L[i][2], L[i][3]) for i in range(2*N)]  # (type, origIndex)
    
    # A helper function to check if a circular rotation r of S is balanced
    # and if so, build the matching (R array of length N).
    def check_rotation_and_build(r):
        stack = []
        R = [-1]*(N+1)  # 1-based for P; R[p] = q
        # We'll go through 2N steps in the rotated order
        for i in range(2*N):
            idx_in_L = (r + i) % (2*N)
            c, orig_id = data_for_reconstruction[idx_in_L]  # '(' or ')', original ID
            if c == '(':
                stack.append(idx_in_L)
            else:
                # c == ')'
                if not stack:
                    return None  # not balanced
                openPos = stack.pop()
                # We have a pair (openPos, idx_in_L).
                # openPos is a '(' => P, idx_in_L is a ')' => Q
                _, pID = data_for_reconstruction[openPos]  # (type='(', pID)
                _, qID = data_for_reconstruction[idx_in_L] # (type=')', qID)
                R[pID] = qID
        if stack:
            return None  # not balanced (some '(' unmatched)
        return R

    # Try all rotations r in [0..2N-1].
    # If we find one that yields a balanced parentheses matching, we output it.
    for r in range(2*N):
        res = check_rotation_and_build(r)
        if res is not None:
            # We found a valid matching. Now print it.
            # res[pID] = qID for pID in 1..N
            # We need R_1..R_N in order
            # res is size N+1 but index 0 is unused
            print(' '.join(map(str, res[1:])))
            return

    # If no rotation worked, print -1.
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()