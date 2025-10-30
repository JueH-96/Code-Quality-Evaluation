import sys
import threading
def main():
    import sys
    from functools import cmp_to_key
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # floor_sum from AtCoder library:
    # sum_{i = 0..n-1} floor((a*i + b)/m)
    def floor_sum(n, m, a, b):
        res = 0
        # n >= 0, m > 0, a,b >= 0
        while True:
            if a >= m:
                # contribute sum_{i=0..n-1} (a//m)*i
                res += (n - 1) * n * (a // m) // 2
                a %= m
            if b >= m:
                # contribute (b//m) * n
                res += n * (b // m)
                b %= m
            # now a < m, b < m
            y_max = (a * n + b) // m
            if y_max == 0:
                break
            # x_max is smallest x so that floor((a*x + b)/m) = y_max
            x_max = y_max * m - b
            # number of i in [0..n-1] with a*i + b >= y_max*m
            # those i >= ceil(x_max/a)
            # count = n - ceil(x_max/a)
            cnt = n - ( (x_max + a - 1) // a )
            res += cnt * y_max
            # recurse on smaller problem
            # new params: n' = y_max, m' = a, a' = m, b' = (a - x_max % a) % a
            n, m, a, b = y_max, a, m, (a - x_max % a) % a
        return res

    # comparator for sorting by slope = A/B ascending
    def cmp_slope(u, v):
        # compare u[0]/u[1] < v[0]/v[1]
        # i.e. u.A * v.B < v.A * u.B
        au, bu = u[0], u[1]
        av, bv = v[0], v[1]
        # use difference
        uv = au * bv - av * bu
        if uv < 0:
            return -1
        elif uv > 0:
            return 1
        else:
            return 0

    T = int(input())
    out_lines = []
    for _ in range(T):
        line = input().strip()
        if not line:
            line = input().strip()
        N = int(line)
        A = [0]*N
        B = [0]*N
        C = [0]*N
        X_end = None
        for i in range(N):
            ai, bi, ci = map(int, input().split())
            A[i], B[i], C[i] = ai, bi, ci
            # compute D = Ci - Bi - 1
            Di = ci - bi - 1
            # Xi = floor(Di / ai)
            xi = Di // ai
            if X_end is None or xi < X_end:
                X_end = xi
        # if no valid x >= 1
        if X_end is None or X_end < 1:
            out_lines.append("0")
            continue

        # collect lines
        lines = [(A[i], B[i], C[i]) for i in range(N)]
        # sort by slope Ai/Bi ascending
        lines.sort(key=cmp_to_key(cmp_slope))

        # remove duplicate slopes (keep only one with smallest intercept (Ci-1)/Bi)
        uniq = []
        for (ai, bi, ci) in lines:
            if not uniq:
                uniq.append((ai, bi, ci))
            else:
                aj, bj, cj = uniq[-1]
                # if same slope ai/bi == aj/bj
                if ai * bj == aj * bi:
                    # compare intercept (ci-1)/bi vs (cj-1)/bj
                    # keep the one with smaller intercept
                    # (ci-1)*bj < (cj-1)*bi
                    if (ci - 1) * bj < (cj - 1) * bi:
                        uniq[-1] = (ai, bi, ci)
                else:
                    uniq.append((ai, bi, ci))
        # build lower envelope (convex hull trick)
        hull = []
        start = []  # start[j] = (num, den) rational where hull[j] becomes active; start[0]=None
        for (ai, bi, ci) in uniq:
            c1 = ci - 1
            # pop while last line is never active
            while hull:
                aj, bj, cj = hull[-1]
                c1j = cj - 1
                # intersection x0 between (ai,bi,c1) and (aj,bj,c1j):
                # x0 = ((c1)*bj - (c1j)*bi) / (ai*bj - aj*bi)
                U = c1 * bj - c1j * bi
                V = ai * bj - aj * bi
                # if last start is None => last active from -inf, so x0 > -inf always
                if start[-1] is None:
                    break
                # compare x0 <= start_last ?
                num_s, den_s = start[-1]
                # x0 <= num_s/den_s <=> U/V <= num_s/den_s <=> U*den_s <= num_s*V
                # Note: V > 0 because slopes strictly increasing
                if U * den_s <= num_s * V:
                    hull.pop()
                    start.pop()
                else:
                    break
            # now decide start for new line
            if not hull:
                start.append(None)
            else:
                aj, bj, cj = hull[-1]
                c1j = cj - 1
                U = c1 * bj - c1j * bi
                V = ai * bj - aj * bi
                # new line active for x > U/V
                start.append((U, V))
            hull.append((ai, bi, ci))

        # compute L_j (first integer x where hull[j] is active)
        K = len(hull)
        L = [0] * K
        for j in range(K):
            if start[j] is None:
                L[j] = 1
            else:
                num_s, den_s = start[j]
                # floor(num_s/den_s) in Python is num_s//den_s for positive/negative
                xfl = num_s // den_s
                lj = xfl + 1
                if lj < 1:
                    lj = 1
                L[j] = lj

        # sum up contributions
        ans = 0
        X = X_end
        for j in range(K):
            l = L[j]
            if l > X:
                break
            if j + 1 < K:
                r = L[j+1] - 1
            else:
                r = X
            if r > X:
                r = X
            if l > r:
                continue
            ai, bi, ci = hull[j]
            # D = Ci - Bi - 1
            D = ci - bi - 1
            # segment length
            n = r - l + 1
            # prepare floor_sum on transformed u = X - x
            # b_hat = D - ai*r
            b_hat = D - ai * r
            # sum floors_n = sum floor((D - ai*x)/bi) for x in [l..r]
            # transform to sum_{t=0..n-1} floor((ai*t + b_hat)/bi)
            s = floor_sum(n, bi, ai, b_hat)
            # sum f(x) = (#points) + sum floors_n
            ans += n + s

        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()