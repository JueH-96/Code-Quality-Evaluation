#!/usr/bin/env python3
import sys
import threading
def main():
    import sys, math
    data = sys.stdin
    T = int(data.readline())
    out = []
    # floor_sum from AtCoder library, but extended for negative a,b
    def floor_sum(n, m, a, b):
        # sum_{i=0..n-1} floor((a*i + b)/m)
        res = 0
        # handle negative a,b by decomposition
        if a < 0:
            # a = a0 + k*m, where a0 = a mod m
            k = (-a + m - 1)//m
            a += k*m
            # floor((a*i + b)/m) = floor((a0*i + b)/m) - k*i
            res -= k*(n-1)*n//2
        if b < 0:
            k = (-b + m - 1)//m
            b += k*m
            # floor((a*i + b)/m) = floor((a*i + b0)/m) - k
            res -= k*n
        # now a,b >= 0
        if a >= m:
            res += (a//m)*(n-1)*n//2
            a %= m
        if b >= m:
            res += (b//m)*n
            b %= m
        # main recursion
        def _fs(n, m, a, b):
            # assume 0 <= a < m, 0 <= b < m
            if n == 0: return 0
            if a == 0:
                return (b//m) * n
            # if a*n + b < m then all floors zero
            if a*n + b < m:
                return 0
            # otherwise use reciprocity
            y_max = (a*n + b)//m
            x_max = (m*y_max - b + a - 1)//a
            # sum floors = (n)* (y_max) - sum_{j=0..y_max-1} ceil((m*j - b)/a)
            # via swapping
            tmp = _fs(y_max, a, m, (a - b % a) % a)
            return n*y_max - tmp
        res += _fs(n, m, a, b)
        return res

    for _ in range(T):
        N = int(data.readline())
        lines = []
        Xmax = None
        for i in range(N):
            A,B,C = map(int, data.readline().split())
            # x < C/A => x <= floor((C-1)/A)
            xi = (C-1)//A
            if Xmax is None or xi < Xmax:
                Xmax = xi
            lines.append((A,B,C))
        if Xmax is None or Xmax < 1:
            out.append("0")
            continue
        # build lower envelope of f(x) = (C - A x) / B
        # sort by slope = -A/B increasing => by A/B decreasing
        lines.sort(key=lambda t: (t[0]*1.0/t[1]), reverse=True)
        # remove duplicates slopes, keep minimal intercept C/B
        filtered = []
        for A,B,C in lines:
            if not filtered:
                filtered.append((A,B,C))
            else:
                A0,B0,C0 = filtered[-1]
                if A*B0 == A0*B:
                    # same slope, keep one with smaller C/B => smaller C*B0
                    if C*B0 < C0*B:
                        filtered[-1] = (A,B,C)
                else:
                    filtered.append((A,B,C))
        # hull with start_x
        hull = []
        # compute intersection x of i and j
        def intersect(i, j):
            A1,B1,C1 = i; A2,B2,C2 = j
            num = C1*B2 - C2*B1
            den = A1*B2 - A2*B1
            # den >0 because slopes sorted
            return num/den
        for line in filtered:
            while hull:
                if len(hull) == 1:
                    x0 = intersect(hull[-1][0], line)
                    if x0 <= hull[-1][1]:
                        hull.pop()
                    else:
                        break
                else:
                    x0 = intersect(hull[-1][0], line)
                    if x0 <= hull[-1][1]:
                        hull.pop()
                    else:
                        break
            if not hull:
                # start at -inf
                hull.append((line, -1e30))
            else:
                x0 = intersect(hull[-1][0], line)
                hull.append((line, x0))
        # now hull is list of (line, start_x)
        total = 0
        # iterate hull segments
        for idx in range(len(hull)):
            (A,B,C), sx = hull[idx]
            # segment x from L = max(ceil(sx),1) to R = next_sx_ceil-1 or Xmax
            L = max(1, math.floor(sx)+1)
            if idx+1 < len(hull):
                nx = hull[idx+1][1]
                R = min(Xmax, math.floor(nx))
            else:
                R = Xmax
            if L > R:
                continue
            # also need ensure y_max >= 1: floor((C - A x -1)/B) >= 1
            # i.e. (C - A x -1) >= B => x <= floor((C -1 - B)/A)
            x_lim = (C -1 - B)//A
            if x_lim < L:
                continue
            if R > x_lim:
                R = x_lim
            if L > R:
                continue
            # sum g(x) for x in [L..R]: g(x) = floor((C - A x -1)/B)
            # define u = C-1
            u = C-1
            # we want sum_{x=L..R} floor((u - A x)/B)
            # let n = R-L+1, i from 0..n-1, x = L + i
            # floor((u - A*(L+i))/B) = floor(( -A*i + (u - A*L) ) / B)
            n = R - L + 1
            a = -A
            b = u - A*L
            # decompose floor((a*i + b)/B)
            # floor((a*i+b)/B) = floor((a0*i + b0)/B) + (a//B)*i + (b//B)
            a_div = a//B
            b_div = b//B
            a0 = a - a_div*B
            b0 = b - b_div*B
            # sum = sum floor((a0*i + b0)/B) + a_div * sum i + n*b_div
            s0 = floor_sum(n, B, a0, b0)
            si = n*(n-1)//2
            segsum = s0 + a_div * si + n * b_div
            total += segsum
        out.append(str(total))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()