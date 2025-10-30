import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    # dp[k][a] = minimum total saltiness achievable by picking k items with total sweetness exactly a
    INF = 10**18
    # We need up to N items and sweetness up to X
    dp = [[INF] * (X+1) for _ in range(N+1)]
    dp[0][0] = 0
    # Process each dish
    for A, B in items:
        # iterate counts downwards so we don't reuse same item twice
        for k in range(N, 0, -1):
            # sweetness sum downwards
            row_prev = dp[k-1]
            row_cur = dp[k]
            for s in range(X, A-1, -1):
                prev_salt = row_prev[s - A]
                if prev_salt + B < row_cur[s]:
                    row_cur[s] = prev_salt + B
    # Now find the maximum k for which there's some sweetness <= X
    # with saltiness <= Y
    for k in range(N, -1, -1):
        best = min(dp[k])  # smallest salt among all sweetness sums
        if best <= Y:
            print(k)
            return

if __name__ == "__main__":
    main()