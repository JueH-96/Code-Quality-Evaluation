import sys
def main():
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    # Sort by sweetness (A) ascending to reduce DP work
    items.sort(key=lambda x: x[0])
    INF = 10**9
    # dp[k][a] = minimal total saltiness achievable by picking k dishes
    # whose total sweetness is exactly a (or INF if not possible).
    dp = [[INF] * (X+1) for _ in range(N+1)]
    dp[0][0] = 0

    for A, B in items:
        lim = X - A
        # iterate k downward so we don't reuse the same item multiple times
        for k in range(N-1, -1, -1):
            prev = dp[k]
            curr = dp[k+1]
            # for each sweetness sum "a" we can extend by this item
            # new sweetness = a + A, new saltiness = prev[a] + B
            # we only care about a <= lim so that a+A <= X
            for a in range(lim, -1, -1):
                sb = prev[a] + B
                # if it's an improvement, take it
                if sb < curr[a + A]:
                    curr[a + A] = sb

    # Find the largest M such that there is some a<=X with dp[M][a] <= Y
    M = 0
    for k in range(N, -1, -1):
        row = dp[k]
        # scan saltiness values for this k
        ok = False
        for sb in row:
            if sb <= Y:
                ok = True
                break
        if ok:
            M = k
            break

    # If we can take M dishes within both constraints, then we can arrange
    # them first and then take one more (which may push us over), so answer = M+1,
    # but cannot exceed N.
    ans = M + 1 if M < N else N
    print(ans)

if __name__ == "__main__":
    main()