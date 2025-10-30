import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp0 = best sum picking an even number of elements so far
    # dp1 = best sum picking an odd number of elements so far
    # We only allow picks at odd positions when going from dp0->dp1,
    # and picks at even positions when going from dp1->dp0.
    NEG_INF = -10**30
    dp0, dp1 = 0, NEG_INF

    # iterate positions 1..N
    for i, ai in enumerate(A, start=1):
        if (i & 1) == 1:
            # at odd position, we may pick it as the next (odd-numbered) element
            # transition dp0 -> dp1
            v = dp0 + ai
            if v > dp1:
                dp1 = v
        else:
            # at even position, we may pick it as the next (even-numbered) element
            # transition dp1 -> dp0
            v = dp1 + ai
            if v > dp0:
                dp0 = v

    # final pick count K must have same parity as N
    # if N even we want dp0, else dp1
    if (N & 1) == 0:
        print(dp0)
    else:
        print(dp1)

if __name__ == "__main__":
    main()