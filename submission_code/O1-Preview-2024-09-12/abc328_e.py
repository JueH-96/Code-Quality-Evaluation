# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u - 1, v - 1, w))

    min_cost = K + 1

    from itertools import combinations
    comb_edges = list(combinations(range(M), N - 1))

    for comb in comb_edges:
        parent = [i for i in range(N)]

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        is_cycle = False
        sum_weights = 0
        components = N

        for idx in comb:
            u, v, w = edges[idx]
            fu = find(u)
            fv = find(v)
            if fu == fv:
                is_cycle = True
                break
            else:
                parent[fu] = fv
                components -= 1
                sum_weights += w

        if not is_cycle and components == 1:
            cost = sum_weights % K
            if cost < min_cost:
                min_cost = cost

    print(min_cost)

threading.Thread(target=main).start()