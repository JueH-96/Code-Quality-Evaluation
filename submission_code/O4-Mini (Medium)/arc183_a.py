import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    from math import factorial

    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    # Total length
    R = N * K
    # Compute total number of sequences S0 = (R)! / (K!^N)
    # Use Python's fast C-implemented factorial
    facR = factorial(R)
    facK = factorial(K)
    S = facR // (facK ** N)
    # target position T = floor((S+1)/2)
    T = (S + 1) // 2

    # counts of each value remaining
    cnt = [0] + [K] * N

    res = []
    # current total sequences count
    S_cur = S
    rem = R
    # Build sequence one by one
    for pos in range(R):
        # try each value in lex order
        for v in range(1, N+1):
            c = cnt[v]
            if c == 0:
                continue
            # number of sequences if we pick v now:
            # S_i = S_cur * c / rem
            # rem >= 1, integer division exact
            Si = (S_cur * c) // rem
            if T <= Si:
                # pick v
                res.append(str(v))
                # update
                cnt[v] = c - 1
                S_cur = Si
                rem -= 1
                # T stays as is
                break
            else:
                T -= Si
        else:
            # should never happen
            pass

    # output
    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()