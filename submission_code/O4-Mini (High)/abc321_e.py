import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        T = int(next(it))
    except StopIteration:
        return
    out = []
    # Process each test case
    for _ in range(T):
        N = int(next(it))
        X = int(next(it))
        K = int(next(it))
        # depth of X = floor(log2(X))
        dX = X.bit_length() - 1
        # maximum h to go up
        H = K if K < dX else dX
        Np1 = N + 1
        total = 0
        # h = 0 case: stay at X and go down K
        # prev_u will track u_{h-1}; initialize to u0 = X
        prev_u = X
        # compute for h = 0
        D = K
        # count1 = number of descendants of prev_u at depth D
        # we only compute if prev_u*(2^D) <= N, i.e. prev_u <= N >> D
        if prev_u <= (N >> D):
            L = prev_u << D
            # (prev_u+1)<<D is R+1
            t = (prev_u + 1) << D
            # count of nodes = min(N+1, t) - L
            if t <= Np1:
                count1 = t - L
            else:
                count1 = Np1 - L
        else:
            count1 = 0
        total = count1
        # h >= 1 cases
        for h in range(1, H+1):
            # w = u_{h-1}
            w = prev_u
            # u = u_h = X >> h = w >> 1
            u = w >> 1
            D = K - h
            # count1 for u at depth D
            if u <= (N >> D):
                L = u << D
                t = (u + 1) << D
                if t <= Np1:
                    count1 = t - L
                else:
                    count1 = Np1 - L
            else:
                count1 = 0
            # count2 forbidding subtree through w at depth D-1
            dp = D - 1
            if dp >= 0 and w <= (N >> dp):
                L2 = w << dp
                t2 = (w + 1) << dp
                if t2 <= Np1:
                    count2 = t2 - L2
                else:
                    count2 = Np1 - L2
            else:
                count2 = 0
            total += (count1 - count2)
            prev_u = u
        out.append(str(total))
    sys.stdout.write("
".join(out))

# call main
main()