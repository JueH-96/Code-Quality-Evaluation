import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M, Q = map(int, input().split())
    arcs_hills = []
    arcs_vals = []
    # Read persons and split into hill/valley arcs
    for idx in range(1, M+1):
        s, t = map(int, input().split())
        if s < t:
            # hill
            arcs_hills.append((s, t, idx))
        else:
            # valley
            arcs_vals.append((t, s, idx))
    # cross_prev[j] = largest i<j such that i and j cross
    cross_prev = [0] * (M+1)
    def detect(arcs):
        # sort by l asc, and for equal l by r desc
        arcs.sort(key = lambda x: (x[0], -x[1]))
        stack = []
        for l, r, idx in arcs:
            # pop any previous arcs whose r < current r
            while stack and stack[-1][1] < r:
                top_l, top_r, top_idx = stack.pop()
                # check strict crossing: top.l < cur.l < top.r < cur.r
                # i.e. top_l < l < top_r (we already know top_r < r)
                if top_l < l and top_r > l:
                    # record crossing pair (i,j) with i<j
                    if top_idx < idx:
                        i, j = top_idx, idx
                    else:
                        i, j = idx, top_idx
                    # update last crossing for j
                    if cross_prev[j] < i:
                        cross_prev[j] = i
            # push current arc
            stack.append((l, r, idx))

    # detect among hills and among valleys
    detect(arcs_hills)
    detect(arcs_vals)

    # build prefix array of "last bad index"
    preBad = [0] * (M+1)
    for j in range(1, M+1):
        # preBad[j] = max(preBad[j-1], cross_prev[j])
        v = cross_prev[j]
        if v > preBad[j-1]:
            preBad[j] = v
        else:
            preBad[j] = preBad[j-1]

    out = []
    for _ in range(Q):
        L, R = map(int, input().split())
        # no crossing inside [L,R] iff preBad[R] < L
        if preBad[R] < L:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()