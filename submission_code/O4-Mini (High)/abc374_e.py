import sys
def main():
    input = sys.stdin.readline
    N, X = map(int, input().split())
    A = [0]*N
    P = [0]*N
    B = [0]*N
    Q = [0]*N
    for i in range(N):
        a, p, b, q = map(int, input().split())
        A[i], P[i], B[i], Q[i] = a, p, b, q

    def feasible(M):
        # returns True if we can achieve capacity >= M on all processes within budget X
        total_cost = 0
        for i in range(N):
            ai, pi, bi, qi = A[i], P[i], B[i], Q[i]
            # if M == 0, cost is zero for this process
            if M == 0:
                continue

            # best cost for pure S or pure T
            # ceil division helpers
            cost_s = ((M + ai - 1) // ai) * pi
            cost_t = ((M + bi - 1) // bi) * qi
            best = cost_s if cost_s < cost_t else cost_t

            # try mixing small counts of S machines (0..bi)
            # by theory it's enough to try x up to bi
            for x in range(bi + 1):
                cap_from_s = ai * x
                if cap_from_s >= M:
                    cost = x * pi
                else:
                    rem = M - cap_from_s
                    y = (rem + bi - 1) // bi
                    cost = x * pi + y * qi
                if cost < best:
                    best = cost

            total_cost += best
            if total_cost > X:
                return False
        return total_cost <= X

    # set an upper bound for search: max capacity per cost is at most max(A,B),
    # so over budget X you can't exceed max(A,B)*X.
    maxAB = 0
    for i in range(N):
        if A[i] > maxAB: maxAB = A[i]
        if B[i] > maxAB: maxAB = B[i]
    lo, hi = 0, maxAB * X

    ans = 0
    # binary search for the maximum M such that feasible(M) is True
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    print(ans)

# call main to execute
main()