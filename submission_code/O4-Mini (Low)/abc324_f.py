import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # adjacency list: for each u, list of (v, beauty, cost)
    edges = [[] for _ in range(N+1)]
    max_ratio = 0.0
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        edges[u].append((v, b, c))
        # track an upper bound for ratio
        max_ratio = max(max_ratio, b / c)

    def can_achieve(R):
        # DP over the DAG in topological order 1..N
        # dp[u] = max total (b - R*c) achievable to u
        dp = [-1e18] * (N+1)
        dp[1] = 0.0
        for u in range(1, N+1):
            val_u = dp[u]
            if val_u < -1e17:
                # unreachable
                continue
            for v, b, c in edges[u]:
                score = val_u + (b - R * c)
                if score > dp[v]:
                    dp[v] = score
        return dp[N] >= 0.0

    # binary search on R
    lo, hi = 0.0, max_ratio
    for _ in range(80):
        mid = (lo + hi) / 2
        if can_achieve(mid):
            lo = mid
        else:
            hi = mid

    # lo is our answer
    print("{:.15f}".format(lo))


if __name__ == "__main__":
    main()