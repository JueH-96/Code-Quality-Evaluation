# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u,v = map(int, sys.stdin.readline().split())
        u -=1
        v -=1
        edges[u].append(v)
        edges[v].append(u)
    W = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    NODES = N
    f = [1]*NODES  # f[i] is the maximum number of operations for a piece starting at i
    # Prepare list of neighbors with W_y < W_i
    W_v_max = max(W)
    # We will proceed in decreasing order of W_i
    vertices_by_W = [(W[i], i) for i in range(N)]
    vertices_by_W.sort()
    for idx in range(N):
        v = vertices_by_W[idx][1]
        W_v = W[v]
        neighbors = []
        for y in edges[v]:
            if W[y] < W_v:
                neighbors.append((W[y], f[y]))
        if not neighbors:
            continue
        deg_v = len(neighbors)
        max_fv = 1
        if deg_v <= 15:
            # Enumerate all subsets
            from itertools import combinations
            max_total_value = 0
            items = neighbors
            for mask in range(1, 1 << deg_v):
                total_weight = 0
                total_value = 0
                for i in range(deg_v):
                    if mask & (1 << i):
                        w_y, f_y = neighbors[i]
                        total_weight += w_y
                        if total_weight >= W_v:
                            break
                        total_value += f_y
                else:
                    if total_value > max_total_value:
                        max_total_value = total_value
            f[v] = 1 + max_total_value
        else:
            # Use approximate DP
            K = 100  # Max number of items to consider
            # Sort neighbors by decreasing f(y)
            neighbors.sort(key=lambda x: -x[1])
            items = neighbors[:K]
            capacity = W_v -1
            if capacity <= 0:
                continue
            dp = [0]*(capacity +1)
            for w_y, f_y in items:
                weight = w_y
                value = f_y
                for w in range(capacity, weight -1, -1):
                    if dp[w - weight] + value > dp[w]:
                        dp[w] = dp[w - weight] + value
            max_total_value = max(dp)
            f[v] = 1 + max_total_value
    total_operations = 0
    for i in range(N):
        total_operations += A[i] * f[i]
    print(total_operations)

threading.Thread(target=main,).start()