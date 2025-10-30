import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, X = map(int, input().split())
    A = [0]*N
    P = [0]*N
    B = [0]*N
    Q = [0]*N
    for i in range(N):
        ai, pi, bi, qi = map(int, input().split())
        A[i], P[i], B[i], Q[i] = ai, pi, bi, qi

    # Check if we can achieve capacity = C within budget X
    def feasible(C):
        total_cost = 0
        for i in range(N):
            ai, pi, bi, qi = A[i], P[i], B[i], Q[i]
            best = 10**30
            # Try using u units of machine S_i for u in [0 .. bi-1]
            # that's enough to cover all residue classes modulo bi
            max_u = min((C + ai - 1)//ai, bi - 1)
            for u in range(max_u+1):
                cap_from_s = ai * u
                rem = C - cap_from_s
                if rem <= 0:
                    cost = pi * u
                else:
                    # need v units of T_i
                    v = (rem + bi - 1)//bi
                    cost = pi * u + qi * v
                if cost < best:
                    best = cost
            # Also, in case all from T is better (u=0 covered above),
            # or all from S (when rem<=0 for some u).
            total_cost += best
            if total_cost > X:
                return False
        return total_cost <= X

    # Binary search on C. First find an upper bound where feasible is False.
    lo = 0
    hi = 1
    # feasible(0) == True always
    while feasible(hi):
        hi <<= 1
        # safeguard
        if hi > 10**15:
            break

    # Now lo is feasible, hi possibly not. Binary search for top feasible.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == "__main__":
    main()