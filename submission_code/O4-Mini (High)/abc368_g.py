import sys
import threading
def main():
    import sys, bisect
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # Ceil‐divide a/b, b>0, works even for a<0:
    def ceil_div(a,b):
        return (a + b - 1)//b

    # Given a list of raw lines (m,b), build the upper hull
    # on x>=0 of max(m*x + b).  We assume b lists fits in memory.
    def build_hull(lines):
        # 1) sort by slope m ascending, drop equal‐slope lines keeping only max b
        lines.sort(key=lambda t:(t[0], -t[1]))
        nl = []
        lastm = None
        for (m,b) in lines:
            if m != lastm:
                nl.append((m,b))
                lastm = m
        # 2) build the convex hull
        hull = []   # list of (m,b)
        xs = []     # xs[i] = minimal x>=0 from which hull[i] is best
        for (m,b) in nl:
            if not hull:
                hull.append((m,b))
                xs.append(0)
            else:
                # pop back while new line makes the previous irrelevant
                while True:
                    m0,b0 = hull[-1]
                    # intersect at x0 = ceil((b0 - b)/(m - m0))
                    # for x>=x0 the new line >= old line
                    x0 = ceil_div(b0 - b, m - m0)
                    if x0 <= xs[-1]:
                        hull.pop()
                        xs.pop()
                        if not hull:
                            break
                    else:
                        break
                # now add the new line
                if not hull:
                    hull.append((m,b))
                    xs.append(0)
                else:
                    m0,b0 = hull[-1]
                    x0 = ceil_div(b0 - b, m - m0)
                    if x0 < 0:
                        x0 = 0
                    hull.append((m,b))
                    xs.append(x0)
        return hull, xs

    # Evaluate the hull at a given x>=0
    def eval_hull(hull, xs, x):
        i = bisect.bisect_right(xs, x) - 1
        m,b = hull[i]
        return m*x + b

    # read input
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())

    # build segment‐tree
    # size M = next power of two >= N
    M = 1
    while M < N:
        M <<= 1
    size = 2*M
    seg = [None] * size

    # build leaves
    for i in range(N):
        ai = A[i]
        bi = B[i]
        if bi == 1:
            raw = [(1, ai)]
        else:
            raw = [(1, ai), (bi, 0)]
        seg[M+i] = build_hull(raw)
    # empty leaves beyond N
    for i in range(N, M):
        seg[M+i] = ([(1,0)], [0])

    # build internal nodes
    for p in range(M-1, 0, -1):
        left_h, left_x = seg[2*p]
        right_h, right_x = seg[2*p+1]
        # compose: f = f_r ∘ f_l  ==> max_{l in left, r in right} (m_r*m_l*x + m_r*b_l + b_r)
        raw = []
        for m_l,b_l in left_h:
            for m_r,b_r in right_h:
                raw.append((m_r*m_l, m_r*b_l + b_r))
        seg[p] = build_hull(raw)

    out = []
    for _ in range(Q):
        line = input().split()
        t = int(line[0])
        if t == 1:
            # update A[i]
            i = int(line[1]) - 1
            x = int(line[2])
            A[i] = x
            # rebuild leaf
            idx = M + i
            if B[i] == 1:
                raw = [(1, x)]
            else:
                raw = [(1, x), (B[i], 0)]
            seg[idx] = build_hull(raw)
            # climb
            idx //= 2
            while idx:
                left_h, left_x = seg[2*idx]
                right_h, right_x = seg[2*idx+1]
                raw = []
                for m_l,b_l in left_h:
                    for m_r,b_r in right_h:
                        raw.append((m_r*m_l, m_r*b_l + b_r))
                seg[idx] = build_hull(raw)
                idx //= 2

        elif t == 2:
            # update B[i]
            i = int(line[1]) - 1
            x = int(line[2])
            B[i] = x
            # rebuild leaf
            idx = M + i
            if x == 1:
                raw = [(1, A[i])]
            else:
                raw = [(1, A[i]), (x, 0)]
            seg[idx] = build_hull(raw)
            idx //= 2
            while idx:
                left_h, left_x = seg[2*idx]
                right_h, right_x = seg[2*idx+1]
                raw = []
                for m_l,b_l in left_h:
                    for m_r,b_r in right_h:
                        raw.append((m_r*m_l, m_r*b_l + b_r))
                seg[idx] = build_hull(raw)
                idx //= 2

        else:
            # query type 3: compute DP on [l,r]
            l = int(line[1]) - 1
            r = int(line[2]) - 1
            L = l + M
            R = r + M
            left_list = []
            right_list = []
            while L <= R:
                if (L & 1):
                    left_list.append(L)
                    L += 1
                if not (R & 1):
                    right_list.append(R)
                    R -= 1
                L >>= 1; R >>= 1
            seq = left_list + right_list[::-1]
            v = 0
            for p in seq:
                h, xs = seg[p]
                v = eval_hull(h, xs, v)
            out.append(str(v))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()