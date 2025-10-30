# YOUR CODE HERE
import sys
import threading

def main():
    import math
    import bisect

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    max_b = 0
    min_c = 1e9
    for _ in range(M):
        u_i, v_i, b_i, c_i = map(int, sys.stdin.readline().split())
        u_i -= 1  # zero-based indexing
        v_i -= 1

        edges.append((u_i, v_i, b_i, c_i))
        max_b = max(max_b, b_i)
        min_c = min(min_c, c_i)

    # Build adjacency list of incoming edges for each node
    adj = [[] for _ in range(N)]
    for u, v, b, c in edges:
        adj[v].append((u, b, c))

    # Binary search over r
    lower = 0.0
    upper = 1e8  # Since beauties and costs upto 1e4, maximum ratio can be upto 1e4
    eps = 1e-7
    iteration = 0
    while upper - lower > eps:
        mid = (lower + upper) / 2.0
        # DP over nodes
        dp = [float('-inf')] * N
        dp[0] = 0.0  # dp[1] = 0

        for v in range(1, N):
            for u, b, c in adj[v]:
                if dp[u] != float('-inf'):
                    val = dp[u] + b - mid * c
                    if val > dp[v]:
                        dp[v] = val

        if dp[N-1] >= 0.0:
            lower = mid
        else:
            upper = mid
        iteration += 1

    answer = (lower + upper) / 2.0
    print("%.16f" % answer)

threading.Thread(target=main).start()