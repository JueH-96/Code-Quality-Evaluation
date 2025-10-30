import sys
import threading
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Build adjacency list for DAG (u < v ensures topological order is 1..N)
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(next(it)); v = int(next(it))
        b = int(next(it)); c = int(next(it))
        adj[u].append((v, b, c))
    # Check function: given lambda, returns True if exists path 1->N with sum(b - lambda*c) >= 0
    def feasible(lam):
        # dp[v]: maximum modified weight to reach v
        # initialize
        dp = [-1e18] * (N+1)
        dp[1] = 0.0
        # process in topological order 1..N
        for u in range(1, N+1):
            du = dp[u]
            if du <= -1e17:
                continue
            for v, b, c in adj[u]:
                val = du + b - lam * c
                # manual inline max
                if val > dp[v]:
                    dp[v] = val
        return dp[N] >= 0.0

    # Binary search for maximum ratio
    lo = 0.0
    hi = 10000.0  # b_i <= 1e4, c_i >=1 => ratio <= 1e4
    # iterate enough times for 1e-9 precision
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    # Print answer
    # Using lo as the best feasible ratio
    sys.stdout.write("{:.16f}
".format(lo))

if __name__ == "__main__":
    main()