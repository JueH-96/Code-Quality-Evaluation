# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    mod = 998244353

    # Build list of nodes with extra edges
    extra_edges = [[] for _ in range(N)]
    reverse_edges = [[] for _ in range(N)]
    for _ in range(M):
        X, Y = map(int, sys.stdin.readline().split())
        X -= 1
        Y -= 1
        extra_edges[X].append(Y)
        reverse_edges[Y].append(X)

    # Nodes with extra edges
    has_extra = [False]*N
    degrees = [1]*N  # Each node has at least the default edge to v+1 mod N
    for i in range(N):
        degrees[i] += len(extra_edges[i])
        if len(extra_edges[i]) > 0:
            has_extra[i] = True

    # Precompute the number of times each node appears in the base path
    # From node 1, base path is nodes (1 + s) % N for s = 0 to K-1
    # For node v, number of times it appears is t[v] = (K - (v - 1) + N - 1) // N
    # Adjust indices since nodes are 0-indexed
    t = [0]*N
    for v in range(N):
        # Positions when node v appears in base path
        first = (v - 1) % N
        total = (K - first + N -1) // N
        t[v] = max(0, total)

    result = 1
    for v in range(N):
        if t[v] > 0 and len(extra_edges[v]) > 0:
            # Total choices at node v is 1 (default edge) + number of extra edges
            choices = len(extra_edges[v]) + 1
            # Total ways contributed by this node is choices^t[v]
            result = result * pow(choices, t[v], mod) % mod

    print(result % mod)
    
threading.Thread(target=main).start()