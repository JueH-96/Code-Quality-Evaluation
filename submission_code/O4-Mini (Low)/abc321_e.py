import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        N = int(next(it)); X = int(next(it)); K = int(next(it))
        cnt = 0
        A = X
        # iterate number of steps we go up: d_up = 0..K
        # but stop when A == 0
        d_up = 0
        while d_up <= K and A > 0:
            d_down = K - d_up
            # total descendants of A at depth d_down within <=N
            # nodes from [A*2^d_down, (A+1)*2^d_down - 1]
            # compute left and right ends
            # if left > N, none
            if d_down < 64:  # beyond that left > N for N <=1e18
                left = A << d_down
                if left <= N:
                    right = ((A + 1) << d_down) - 1
                    if right > N:
                        right = N
                    total = right - left + 1
                else:
                    total = 0
            else:
                total = 0
            if total:
                if d_up == 0 or d_down == 0:
                    # for d_up=0 (direct subtree) or d_down=0 (ancestor itself),
                    # all counted
                    cnt += total
                else:
                    # need to subtract those in branch that goes back toward X
                    # child B of A on path to X
                    B = X >> (d_up - 1)
                    # subtree of B at depth d_down-1 under A
                    dd2 = d_down - 1
                    if dd2 < 64:
                        left2 = B << dd2
                        if left2 <= N:
                            right2 = ((B + 1) << dd2) - 1
                            if right2 > N:
                                right2 = N
                            bad = right2 - left2 + 1
                        else:
                            bad = 0
                    else:
                        bad = 0
                    cnt += (total - bad)
            # move one up
            A >>= 1
            d_up += 1
        out.append(str(cnt))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()