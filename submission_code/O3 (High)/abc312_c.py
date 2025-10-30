import sys
import bisect

def main() -> None:
    input = sys.stdin.readline

    # read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # sort for binary-search
    A.sort()                 # ascending
    B.sort()                 # ascending  (makes counting ≥X easy)

    # helper to count sellers / buyers for a given price X
    # sellers: A_i ≤ X   -> bisect_right
    # buyers : B_i ≥ X   -> M - bisect_left
    def ok(x: int) -> bool:
        sellers = bisect.bisect_right(A, x)
        buyers  = M - bisect.bisect_left(B, x)
        return sellers >= buyers

    # binary search for the minimal X with ok(X) == True
    lo = 0                                      # definitely NG (buyers = M > 0, sellers = 0)
    hi = max(max(A), max(B)) + 1                # definitely OK (all sellers ready, no buyers)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid

    print(hi)

if __name__ == "__main__":
    main()