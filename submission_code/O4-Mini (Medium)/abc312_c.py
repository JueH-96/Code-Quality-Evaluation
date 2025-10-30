import sys
import threading

def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]

    A.sort()
    B.sort()

    # Define condition: for price X,
    # count of sellers with A_i <= X >= count of buyers with B_i >= X
    def condition(X):
        # sellers who can sell at price X
        sellers = bisect.bisect_right(A, X)
        # buyers who can buy at price X
        # bisect_left gives first B[i] >= X, so M - idx is count of >= X
        buyers = M - bisect.bisect_left(B, X)
        return sellers >= buyers

    # We know condition(0) is false (since B_i >= 1, buyers>0, sellers=0)
    # and condition(max(B)+1) is true (no buyers can pay > max(B)).
    lo = 0
    hi = B[-1] + 1  # hi > max(B), so buyers = 0 => condition true

    # binary search for smallest X such that condition(X) is true
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid

    print(hi)

if __name__ == "__main__":
    main()