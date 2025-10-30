import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); W = int(line[1])
    # sentinel for unreachable states
    NEG_INF = -10**18
    # dp[j] = max happiness for weight exactly j
    dp = [0] + [NEG_INF] * W
    for _ in range(N):
        parts = data.readline().split()
        if not parts:
            parts = data.readline().split()
            if not parts:
                break
        w = int(parts[0]); v = int(parts[1])
        # max number of positive-profit items of this type
        k_max = v // 2
        c = W // w
        if k_max > c:
            k_max = c
        # new dp
        dp2 = [NEG_INF] * (W + 1)
        dp_old = dp  # alias
        # process each residue class mod w
        for r in range(w):
            # T = max t such that r + t*w <= W
            # if r > W, then T = negative, skip
            T = (W - r) // w
            if T < 0:
                continue
            # convex hull: lines of form y = m*x + b, with m increasing
            M = []  # slopes (u)
            B = []  # intercepts (A_u = dp_old[r+u*w] - u^2)
            head = 0
            # iterate t = number of base-steps
            # j = r + t*w is the weight index
            # dp2[j] = max_{0 <= k <= min(t, k_max)} dp_old[r + (t-k)*w] + k*v - k^2
            # transformed with hull trick
            vv = v  # local
            km = k_max
            ww = w
            # loop over t
            for t in range(T + 1):
                j = r + t * ww
                # compute intercept for new line corresponding to u = t
                # A = dp_old[j] - t^2
                # dp_old[j] might be NEG_INF, subtraction safe
                A = dp_old[j] - t * t
                m = t
                # add new line (m, A) with monotonic slopes
                # remove last while last two + new make middle obsolete
                # condition: intersection(l1,l2) >= intersection(l2,new)
                # (b1 - b2)/(m2-m1) >= (b2 - b3)/(m3-m2)
                # --> (b1 - b2)*(m3 - m2) >= (b2 - b3)*(m2 - m1)
                # here l1 = (M[-2],B[-2]), l2 = (M[-1],B[-1]), new = (m,A)
                while len(M) - head >= 2:
                    # indices of last two in M list
                    # careful: M[-1] is last, M[-2] second last
                    m2 = M[-1]; b2 = B[-1]
                    m1 = M[-2]; b1 = B[-2]
                    # new line is (m,b) = (m,A)
                    # check if l2 is obsolete
                    # (b1 - b2)*(m - m2) >= (b2 - A)*(m2 - m1)
                    if (b1 - b2) * (m - m2) >= (b2 - A) * (m2 - m1):
                        M.pop(); B.pop()
                    else:
                        break
                M.append(m); B.append(A)
                # remove lines with u < t - k_max (invalid because k = t-u > k_max)
                bound = t - km
                # head moves right
                # while head < len(M) and M[head] < bound: head += 1
                # Python fast local
                mh = head
                MM = M
                while mh < len(MM) and MM[mh] < bound:
                    mh += 1
                head = mh
                # query best line at x = 2*t - v
                x = (t << 1) - vv
                # advance head while next line gives better value
                # M[head] * x + B[head] vs M[head+1] * x + B[head+1]
                # as x increases over t, monotonic queue query
                # ensure head+1 in range
                MH = M; BH = B
                hh = head
                # localize len for speed
                ln = len(MH)
                # compare two lines
                # while hh+1 < ln and MH[hh+1]*x + BH[hh+1] >= MH[hh]*x + BH[hh]:
                #     hh += 1
                # head = hh
                while hh + 1 < ln:
                    # compute value at head and next
                    # note: Python multiplies ints; overhead minimal
                    if MH[hh+1] * x + BH[hh+1] >= MH[hh] * x + BH[hh]:
                        hh += 1
                    else:
                        break
                head = hh
                # best value from hull
                best = MH[head] * x + BH[head]
                # compute C = -t^2 + v*t = t*(v - t)
                C = t * (vv - t)
                dp2[j] = best + C
        # move to next type
        dp = dp2
    # answer is max happiness for any weight <= W
    ans = 0
    # dp may contain NEG_INF for unreachable; reachable dp >= 0
    for val in dp:
        if val > ans:
            ans = val
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()