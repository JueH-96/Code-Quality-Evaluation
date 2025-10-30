import sys
from math import inf

# -------------------------------------------------------------

def feasible(adj, N, x) -> bool:
    """
    True  : there is a path 1 -> N with ratio >= x
    False : otherwise
    """
    NEG = -1e100
    dp = [NEG] * (N + 1)
    dp[1] = 0.0

    for u in range(1, N + 1):
        du = dp[u]
        if du == NEG:
            continue
        for v, b, c in adj[u]:
            val = du + (b - x * c)
            if val > dp[v]:
                dp[v] = val
    return dp[N] >= 0.0


# -------------------------------------------------------------

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    adj = [[] for _ in range(N + 1)]
    max_ratio = 0.0
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        b = int(next(it))
        c = int(next(it))
        adj[u].append((v, b, c))
        max_ratio = max(max_ratio, b / c)

    lo, hi = 0.0, max_ratio
    for _ in range(60):                # 2⁻⁶⁰  < 10⁻¹⁸
        mid = (lo + hi) * 0.5
        if feasible(adj, N, mid):
            lo = mid
        else:
            hi = mid

    # 15 digits after the decimal point are enough for 1e-9 precision
    print(f"{lo:.15f}")


# -------------------------------------------------------------
if __name__ == "__main__":
    main()