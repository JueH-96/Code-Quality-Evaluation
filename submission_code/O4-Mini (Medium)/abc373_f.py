import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    W = int(next(it))
    dp = [0] * (W + 1)
    NEG_INF = -10**30
    from collections import deque
    for _ in range(N):
        w = int(next(it))
        v = int(next(it))
        old = dp
        # prepare new dp for this group
        dp = [NEG_INF] * (W + 1)
        # process each residue class modulo w
        for r in range(w):
            # collect old dp values for this residue
            d = []
            pos = []
            j = r
            while j <= W:
                d.append(old[j])
                pos.append(j)
                j += w
            M = len(d) - 1
            if M < 0:
                continue
            # we'll compute C[m] for m = 0..M
            # use convex hull trick on lines of form y = slope * x + intercept
            # slope = 2*t, intercept = A[t] = d[t] - t*v - t^2
            hull = deque()
            # helper to test if line l2 is unnecessary between l1 and l3
            def bad(l1, l2, l3):
                # l1: (m1,b1), l2: (m2,b2), l3: (m3,b3)
                m1,b1 = l1
                m2,b2 = l2
                m3,b3 = l3
                # remove l2 if intersection(l1,l2) >= intersection(l1,l3)
                # (b2-b1)/(m1-m2) >= (b3-b1)/(m1-m3)
                return (b2 - b1) * (m1 - m3) >= (b3 - b1) * (m1 - m2)
            # build hull and query as m increases
            # C[m] = m*v - m^2 + max_t [ A[t] + 2*t*m ]
            # where A[t] = d[t] - t*v - t^2
            for m in range(M+1):
                # compute line for t = m
                A = d[m] - m * v - m * m
                line = (2*m, A)
                # add to hull, maintain upper hull
                while len(hull) >= 2 and bad(hull[-2], hull[-1], line):
                    hull.pop()
                hull.append(line)
                # query best line at x = m; since x increases, we can pop from front
                # if next line gives better value
                while len(hull) >= 2:
                    m0, b0 = hull[0]
                    m1, b1 = hull[1]
                    # if hull[1] yields >= at x=m, drop hull[0]
                    if m0 * m + b0 <= m1 * m + b1:
                        hull.popleft()
                    else:
                        break
                # evaluate
                s, b = hull[0]
                best = s * m + b
                C_m = m * v - m * m + best
                # write into dp
                dp[pos[m]] = C_m
        # dp for this type done
    # answer is max happiness with weight <= W
    # dp[W] holds optimum for exactly weight W, but since dp transitions carried over
    # and we always allow k=0, dp[j] is best achievable for weight j exactly.
    # We want max_{j<=W} dp[j], but since adding no items keeps dp[<W] from prev types,
    # dp[W] is the maximum for exactly W. However, user may not fill exactly W.
    # So take max over dp[0..W].
    ans = max(dp)
    print(ans)

if __name__ == "__main__":
    main()