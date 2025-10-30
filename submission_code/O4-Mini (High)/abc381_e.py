#!/usr/bin/env python3
import sys
import threading
def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    S = next(it).strip()
    # Prefix sums of '1' and '2'
    P1 = [0] * (n+1)
    P2 = [0] * (n+1)
    for i, ch in enumerate(S, start=1):
        P1[i] = P1[i-1]
        P2[i] = P2[i-1]
        if ch == '1':
            P1[i] += 1
        elif ch == '2':
            P2[i] += 1
    # Slash positions and x, y, z arrays
    s = []
    # x[i] = #1s before slash at s[i], i.e. P1[s[i]-1]
    # y[i] = #2s up to slash at s[i], i.e. P2[s[i]]
    # z[i] = x[i] + y[i]
    x = []
    y = []
    z = []
    for i, ch in enumerate(S, start=1):
        if ch == '/':
            s.append(i)
            xi = P1[i-1]
            yi = P2[i]
            x.append(xi)
            y.append(yi)
            z.append(xi + yi)
    M = len(s)
    out = []
    # Process queries
    for _ in range(q):
        L = int(next(it))
        R = int(next(it))
        # find slash indices in [L,R]
        li = bisect.bisect_left(s, L)
        ri = bisect.bisect_right(s, R) - 1
        if li > ri:
            out.append('0')
            continue
        a = P1[L-1]
        b = P2[R]
        Sval = a + b
        # find first global crossing index pos0 where z[pos0] >= a+b
        pos0 = bisect.bisect_left(z, Sval)
        # clamp to our [li..]
        if pos0 < li:
            j1 = li
        else:
            j1 = pos0
        # compute max delta
        if j1 <= ri:
            # crossing at j1
            if j1 > li:
                # two candidates
                # candidate at j1-1: U = x[j1-1] - a
                # candidate at j1:   V = b - y[j1]
                d1 = x[j1-1] - a
                d2 = b - y[j1]
                if d1 >= d2:
                    delta = d1
                else:
                    delta = d2
            else:
                # only j1 == li
                delta = b - y[j1]
        else:
            # no crossing in [li..ri], monotonic, max at ri
            delta = x[ri] - a
        # delta >= 0 always when there's a slash
        # answer length = 2*delta + 1
        out.append(str(delta*2 + 1))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()