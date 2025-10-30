import sys
sys.setrecursionlimit(10000)

# ------------- geometry helpers -----------------
def orient(ax, ay, bx, by, cx, cy):
    """cross product of (b-a) x (c-a)   >0 : left,  <0 : right"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


# ------------- recursive construction ------------
def build_matching(P_idx, Q_idx, P, Q, R):
    """
    P_idx : list of indices (in P) that are still unmatched
    Q_idx : list of indices (in Q) that are still unmatched
    R     : result array – R[i] is the index in Q matched to P[i]
    """
    if not P_idx:          # nothing left to do
        return

    # choose a pivot red point  –  the one with the smallest x (then y) is convenient
    p = min(P_idx, key=lambda i: (P[i][0], P[i][1]))
    px, py = P[p]

    # try every blue point until the balancing condition is fulfilled
    for q in Q_idx:
        qx, qy = Q[q]

        # how many reds / blues are to the left of the directed line  p -> q  ?
        reds_left  = 0
        blues_left = 0

        for r in P_idx:
            if r == p:
                continue
            rx, ry = P[r]
            if orient(px, py, qx, qy, rx, ry) > 0:
                reds_left += 1

        for b in Q_idx:
            if b == q:
                continue
            bx, by = Q[b]
            if orient(px, py, qx, qy, bx, by) > 0:
                blues_left += 1

        if reds_left == blues_left:          # balancing achieved!
            R[p] = q                        # store the match

            # partition the remaining points with respect to the line p-q
            P_left, P_right = [], []
            Q_left, Q_right = [], []

            for r in P_idx:
                if r == p:
                    continue
                rx, ry = P[r]
                if orient(px, py, qx, qy, rx, ry) > 0:
                    P_left.append(r)
                else:
                    P_right.append(r)

            for b in Q_idx:
                if b == q:
                    continue
                bx, by = Q[b]
                if orient(px, py, qx, qy, bx, by) > 0:
                    Q_left.append(b)
                else:
                    Q_right.append(b)

            # recurse on the two independent halves
            build_matching(P_left,  Q_left,  P, Q, R)
            build_matching(P_right, Q_right, P, Q, R)
            return

    # According to the known theorem a suitable q always exists, so
    # reaching this point would mean something went wrong.
    raise RuntimeError("Failed to find a non-crossing matching – should be impossible.")


# ------------- main ------------------------------
def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))

    P = [(int(next(it)), int(next(it))) for _ in range(N)]
    Q = [(int(next(it)), int(next(it))) for _ in range(N)]

    R = [-1] * N           # R[i] will hold the index (0-based) of Q matched to P[i]

    build_matching(list(range(N)), list(range(N)), P, Q, R)

    # sanity check – should always succeed
    if any(x == -1 for x in R):
        print(-1)
    else:
        # convert to 1-based indices for output
        print(' '.join(str(x + 1) for x in R))


if __name__ == "__main__":
    main()