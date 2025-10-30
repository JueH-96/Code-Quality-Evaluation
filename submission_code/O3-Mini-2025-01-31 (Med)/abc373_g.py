def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read red points P: coordinates for P_i
    pointsP = []
    for _ in range(n):
        x = int(next(it)); y = int(next(it))
        pointsP.append((x, y))
    # Read blue points Q: coordinates for Q_i
    pointsQ = []
    for _ in range(n):
        x = int(next(it)); y = int(next(it))
        pointsQ.append((x, y))
    
    # Orientation function:
    # For three points p, q, r, the value is positive if r is to left of directed line p->q,
    # negative if to the right.
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    
    # We define a recursion to solve the following:
    # Given a set (state) of red indices and blue indices, find a non crossing pairing.
    # The state is represented by two sorted tuples (red indices, blue indices).
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def solve_state(red_state, blue_state):
        # red_state and blue_state are tuples of indices.
        m = len(red_state)
        if m == 0:
            return ()  # empty matching; we will build a tuple of (red_index, blue_index) pairs.
        if m == 1:
            # Only one possible pairing.
            return ((red_state[0], blue_state[0]),)
        # To break symmetry, choose the red point with smallest x coordinate.
        r_candidate = None
        r_candidate_val = None
        for r in red_state:
            if r_candidate is None or pointsP[r][0] < r_candidate_val:
                r_candidate = r
                r_candidate_val = pointsP[r][0]
        # Try each blue candidate for matching with r_candidate.
        for b in blue_state:
            pos_red = []
            neg_red = []
            for r in red_state:
                if r == r_candidate:
                    continue
                # Compute orientation for red point r relative to the line (pointsP[r_candidate] -> pointsQ[b])
                o = orient(pointsP[r_candidate], pointsQ[b], pointsP[r])
                if o > 0:
                    pos_red.append(r)
                else:
                    neg_red.append(r)
            pos_blue = []
            neg_blue = []
            for bb in blue_state:
                if bb == b:
                    continue
                o = orient(pointsP[r_candidate], pointsQ[b], pointsQ[bb])
                if o > 0:
                    pos_blue.append(bb)
                else:
                    neg_blue.append(bb)
            if len(pos_red) != len(pos_blue):
                continue
            # Recurse on the positive side and the negative side.
            pos_red_t = tuple(sorted(pos_red))
            neg_red_t = tuple(sorted(neg_red))
            pos_blue_t = tuple(sorted(pos_blue))
            neg_blue_t = tuple(sorted(neg_blue))
            left_match = solve_state(pos_red_t, pos_blue_t)
            if left_match is None:
                continue
            right_match = solve_state(neg_red_t, neg_blue_t)
            if right_match is None:
                continue
            current = ((r_candidate, b),)
            return current + left_match + right_match
        return None
    
    red_initial = tuple(sorted(range(n)))
    blue_initial = tuple(sorted(range(n)))
    pairing = solve_state(red_initial, blue_initial)
    if pairing is None:
        sys.stdout.write("-1")
        return
    # Build result: for each red index (corresponding to P_i in order of input)
    # we have a pairing (red, blue). Print R_i = (blue index + 1).
    res = [-1] * n
    for r, b in pairing:
        res[r] = b + 1  # converting to 1-indexed
    sys.stdout.write(" ".join(map(str, res)))
    
if __name__ == '__main__':
    main()