import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    procs = []
    for _ in range(N):
        A = int(next(it)); P = int(next(it))
        B = int(next(it)); Q = int(next(it))
        procs.append((A, P, B, Q))

    from math import gcd

    # Check if we can achieve capacity C with budget X
    def can(C):
        total_cost = 0
        for (A, P, B, Q) in procs:
            # We want min x*P + y*Q s.t. x*A + y*B >= C
            g = gcd(A, B)
            # We'll iterate x in [0 .. (B/g)-1], plus pure S machines
            limit = B // g
            best = 10**30
            # try mixes with small x
            # cost = x*P + ceil(max(0, C - x*A)/B)*Q
            # we only try x from 0 to limit-1
            if limit > 0:
                # local variables for speed
                Ai = A; Bi = B; Pi = P; Qi = Q; Ci = C
                for x in range(limit):
                    # remaining capacity needed
                    rem = Ci - x * Ai
                    if rem <= 0:
                        y = 0
                    else:
                        # ceil division rem/B
                        y = (rem + Bi - 1) // Bi
                    cost = x * Pi + y * Qi
                    if cost < best:
                        best = cost
            # also try pure S: y=0, x = ceil(C/A)
            if A > 0:
                xs = (C + A - 1) // A
                cost_s = xs * P
                if cost_s < best:
                    best = cost_s
            # if best is already exceeding X, we can early reject
            total_cost += best
            if total_cost > X:
                return False
        return total_cost <= X

    # binary search on C
    lo = 0
    hi = 100 * X  # upper bound: max(A_i,B_i)=100, so capacity <=100*X
    # ensure hi is at least 0
    if hi < 0:
        hi = 0

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can(mid):
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == "__main__":
    main()