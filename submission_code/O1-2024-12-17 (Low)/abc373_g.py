def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # Read P_i points
    P = [(int(input_data[2*i+1]), int(input_data[2*i+2])) for i in range(N)]
    # Read Q_i points
    offset = 1 + 2*N
    Q = [(int(input_data[offset + 2*i]), int(input_data[offset + 2*i + 1])) for i in range(N)]

    # We need to pick a reference point T so that
    # no three points (including T) are collinear.
    # We'll choose T well outside the bounding box of given points and
    # check if it is collinear with any pair of our 2N points.
    # If it is, shift T slightly and try again.

    all_points = P + Q

    def collinear(a, b, c):
        # Check if three points a=(x1,y1), b=(x2,y2), c=(x3,y3) are collinear
        # using cross product == 0
        return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) == 0

    # We'll try a small set of candidate T points
    # hoping one is not collinear with any pair of the 2N points.
    candidate_Ts = []
    # Just pick some points far out of range.
    # Problem constraints up to 5000, so let's try negative coords around -10000.
    for dx in range(0, 20):
        candidate_Ts.append((-10000 - dx, -10000 - dx))

    def valid_T(testT):
        # Check if testT is collinear with any pair of points from all_points
        # That means for any pair (p_i, p_j), T, p_i, p_j collinear => not valid
        # We'll do a quick check O((2N)^2) = O(N^2), feasible for N=300
        for i in range(2*N):
            for j in range(i+1, 2*N):
                if collinear(testT, all_points[i], all_points[j]):
                    return False
        return True

    T = None
    for cand in candidate_Ts:
        if valid_T(cand):
            T = cand
            break

    if T is None:
        # In a worst case scenario (extremely unlikely), we failed to find any T.
        # But let's just give up with -1. (Should never realistically happen)
        print(-1)
        return

    # Now we have T. We'll sort all points by angle w.r.t. T.
    # We'll create a list of (angle, type, index)
    # type=0 => from P, type=1 => from Q
    # index is which P_i or Q_i
    arr = []
    for i, (x, y) in enumerate(P):
        ang = math.atan2(y - T[1], x - T[0])
        arr.append((ang, 0, i))
    for i, (x, y) in enumerate(Q):
        ang = math.atan2(y - T[1], x - T[0])
        arr.append((ang, 1, i))

    # Sort by angle
    arr.sort(key=lambda x: x[0])

    # We'll do a stack-based matching:
    # Whenever we see a P, push its index. Whenever we see a Q, pop from stack and match.
    stack = []
    R = [-1]*N  # R[i] = j means P_i matched with Q_j

    for ang, t, idx in arr:
        if t == 0:
            # It's a P
            stack.append(idx)
        else:
            # It's a Q
            if not stack:
                # No available P to match -> no solution
                print(-1)
                return
            p_idx = stack.pop()
            R[p_idx] = idx

    if stack:
        # Some P remain unmatched -> no solution
        print(-1)
        return

    # Otherwise, we have a valid matching stored in R
    # Print R_1, R_2, ..., R_N (1-based index)
    # The problem states: "print R_1, R_2, ..., R_N separated by spaces."
    # Our R array is 0-based Q indices, so we add 1:
    print(' '.join(str(r+1) for r in R))

# Do not forget to call main()
if __name__ == "__main__":
    main()