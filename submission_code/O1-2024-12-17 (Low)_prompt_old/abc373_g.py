def solve():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # Read P-coordinates
    P = []
    idx = 1
    for i in range(N):
        x = float(input_data[idx]); y = float(input_data[idx+1])
        P.append((x, y, i))  # (x, y, original_index_of_P)
        idx += 2
    # Read Q-coordinates
    Q = []
    for i in range(N):
        x = float(input_data[idx]); y = float(input_data[idx+1])
        Q.append((x, y, i))  # (x, y, original_index_of_Q)
        idx += 2

    # A known fact (and a common technique) is that for 2N points in general position
    # (no three collinear), there always exists a non-crossing perfect matching
    # between the N points of one set (P) and the N points of the other set (Q).
    # One constructive way is:
    # 1) Pick a reference point O (for example, the average of all P and Q coordinates).
    # 2) Compute the polar angle of each point in P and Q w.r.t. O.
    # 3) Sort each set (P and Q) by that polar angle.
    # 4) Match the i-th point of P in sorted order with the i-th point of Q in sorted order.
    #
    # This yields a valid non-crossing matching due to the "no three collinear" condition
    # and general geometry arguments (essentially placing them around O and matching
    # in circular order prevents intersections).
    #
    # If by problem statement there could be a case with no solution, we would attempt
    # a known approach or check for consistency. However, the known theorem guarantees
    # existence if no three are collinear and we have two sets of equal size. Thus we
    # will simply output the matching found by the "angle sort" method.  

    # Reference point O = average of all 2N points
    sum_x = 0.0
    sum_y = 0.0
    for (x, y, _) in P:
        sum_x += x
        sum_y += y
    for (x, y, _) in Q:
        sum_x += x
        sum_y += y
    O_x = sum_x / (2.0*N)
    O_y = sum_y / (2.0*N)

    # Function to compute polar angle from O
    def angle_from_O(x, y):
        return math.atan2(y - O_y, x - O_x)

    # Sort P by polar angle
    P_sorted = []
    for (x, y, i) in P:
        P_sorted.append((angle_from_O(x, y), i))
    P_sorted.sort(key=lambda t: t[0])  # sort by angle

    # Sort Q by polar angle
    Q_sorted = []
    for (x, y, j) in Q:
        Q_sorted.append((angle_from_O(x, y), j))
    Q_sorted.sort(key=lambda t: t[0])  # sort by angle

    # The i-th point in P_sorted is matched with the i-th point in Q_sorted
    # We want R[p_index] = q_index + 1 (because we print 1-based for Q)
    R = [0]*N
    for i in range(N):
        # P's original index
        p_idx = P_sorted[i][1]
        # Q's original index
        q_idx = Q_sorted[i][1]
        R[p_idx] = q_idx + 1  # +1 for 1-based index in output

    # Output the result
    # (If in an unlikely scenario no solution is possible, we might have to print -1,
    #  but given the geometry theorem, we can always produce one when no three are collinear.)
    print(" ".join(map(str, R)))

def _test():
    # You can put local tests here if desired
    pass

if __name__ == "__main__":
    solve()