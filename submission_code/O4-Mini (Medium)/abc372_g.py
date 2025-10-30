import sys
import threading
def main():
    import sys
    data = sys.stdin
    T = int(data.readline())
    out = []
    # floor_sum from AtCoder Library
    def floor_sum(n, m, a, b):
        # sum_{i=0..n-1} floor((a*i + b)/m)
        res = 0
        while True:
            if a >= m:
                res += (n - 1) * n * (a // m) // 2
                a %= m
            if b >= m:
                res += n * (b // m)
                b %= m
            # now a, b < m
            y = a * n + b
            if y < m:
                break
            # next
            n = y // m
            b = y % m
            a, m = m, a
        return res

    from functools import cmp_to_key
    def cmp_lines(i, j):
        # compare slope A_i/B_i vs A_j/B_j
        Ai, Bi, Ci = i
        Aj, Bj, Cj = j
        # compare Ai * Bj and Aj * Bi
        v = Ai * Bj - Aj * Bi
        if v != 0:
            return -1 if v < 0 else 1
        # same slope, compare intercept C'/B -> C_i'/B_i < C_j'/B_j?
        # equivalently Ci * Bj vs Cj * Bi
        v2 = Ci * Bj - Cj * Bi
        if v2 != 0:
            return -1 if v2 < 0 else 1
        return 0

    for _ in range(T):
        line = data.readline().split()
        while not line:
            line = data.readline().split()
        N = int(line[0])
        A = [0]*N; B = [0]*N; Cprime = [0]*N
        X_max = None
        for i in range(N):
            ai, bi, ci = map(int, data.readline().split())
            A[i] = ai; B[i] = bi; Cprime[i] = ci - 1
            # compute floor((C' - B)/A)
            # C'-B = (ci-1) - bi
            num = Cprime[i] - bi
            # if num < 0 then no x>=1 is valid -> X_max <1
            xi = num // ai
            if X_max is None or xi < X_max:
                X_max = xi
        if X_max is None or X_max < 1:
            out.append("0")
            continue
        # build list of lines
        lines = [(A[i], B[i], Cprime[i]) for i in range(N)]
        # sort by slope, then intercept
        lines.sort(key=cmp_to_key(cmp_lines))
        # filter same slope, keep only first (smallest intercept)
        filtered = []
        for ai, bi, ci in lines:
            if not filtered:
                filtered.append((ai, bi, ci))
            else:
                ai0, bi0, ci0 = filtered[-1]
                if ai0 * bi != ai * bi0:
                    filtered.append((ai, bi, ci))
                # else same slope, skip (ci/bi >= ci0/bi0 by sort)
        # convex hull for min envelope
        # each node: (A, B, C', x_start)
        hull = []
        # function to compute x_start where new line c becomes <= last line l
        # returns ceil(num/den)
        def intersect_x(c, l):
            # c: (A_c, B_c, Cc), l: (A_l, B_l, Cl)
            A_c, B_c, C_c = c
            A_l, B_l, C_l = l
            # want first x: (C_c'/B_c - A_c/B_c x) <= (C_l'/B_l - A_l/B_l x)
            # => C_c'*B_l - C_l'*B_c <= x*(A_c*B_l - A_l*B_c)
            num = C_c * B_l - C_l * B_c
            den = A_c * B_l - A_l * B_c
            # den > 0 because slopes sorted increasing and not equal
            # x >= num/den
            # ceil:
            if num <= 0:
                return 1
            return (num + den - 1) // den

        for ai, bi, ci in filtered:
            c = (ai, bi, ci)
            # default start at 1
            xs = 1
            # pop while new line overtakes last too early
            while hull:
                l = hull[-1]
                # compute x where c overtakes l
                x0 = intersect_x(c, (l[0], l[1], l[2]))
                if x0 <= l[3]:
                    hull.pop()
                else:
                    xs = x0
                    break
            # if empty hull
            if not hull:
                xs = 1
            # append c with start
            hull.append((ai, bi, ci, xs))
        # now hull segments
        total = 0
        H = len(hull)
        for idx in range(H):
            ai, bi, ci, xs = hull[idx]
            if idx + 1 < H:
                xe = hull[idx+1][3] - 1
            else:
                xe = X_max
            L = xs
            if L < 1:
                L = 1
            if xe > X_max:
                xe = X_max
            if L > xe:
                continue
            # sum x=L..xe floor((ci - ai*x)/bi)
            # let n = xe-L+1
            n = xe - L + 1
            # C' = ci. b0 = ci - ai*L
            b0 = ci - ai * L
            # val_i = floor((b0 - ai*i)/bi) for i=0..n-1
            # we write floor((b0 - ai*i)/bi) = - floor((ai*i + (bi-1 - b0)) / bi)
            b1 = bi - 1 - b0
            # separate b1 = q*bi + r
            # in Python, r = b1 % bi, q = (b1 - r)//bi
            r = b1 % bi
            q = (b1 - r) // bi
            # sumfloor = sum_{i=0..n-1} floor((ai*i + r)/bi) + n*q
            s2 = floor_sum(n, bi, ai, r) + n * q
            seg_sum = -s2
            total += seg_sum
        out.append(str(total))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()