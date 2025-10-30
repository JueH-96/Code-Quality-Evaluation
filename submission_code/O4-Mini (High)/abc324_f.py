import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    edges = []
    max_ratio = 0.0
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))
        # track the maximum possible single‐edge ratio to bound our search
        r = b / c
        if r > max_ratio:
            max_ratio = r

    # Since u < v for all edges, sorting by u gives a topological ordering
    edges.sort(key=lambda x: x[0])

    # Binary search on the answer R
    L, H = 0.0, max_ratio
    for _ in range(60):
        mid = (L + H) * 0.5
        # dp[v] = maximum of sum(b_i - mid*c_i) over all paths 1->v
        dp = [-1e18] * (N + 1)
        dp[1] = 0.0
        for u, v, b, c in edges:
            val = dp[u] + b - mid * c
            if val > dp[v]:
                dp[v] = val
        # if we can get nonnegative total, ratio >= mid is feasible
        if dp[N] >= 0.0:
            L = mid
        else:
            H = mid

    # print answer with sufficient precision
    print("{:.16f}".format(L))

if __name__ == "__main__":
    main()