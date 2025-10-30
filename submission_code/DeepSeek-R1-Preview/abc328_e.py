import sys
from itertools import combinations

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    edges = []
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        w = int(input[idx])
        idx += 1
        edges.append((u, v, w))

    min_mod = K  # Initialize with a value larger than any possible mod

    for combo in combinations(edges, N-1):
        parent = list(range(N + 1))  # 1-based indexing

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u

        for u, v, _ in combo:
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pu] = pv

        # Check if all nodes are connected
        root = find(1)
        connected = True
        for i in range(2, N+1):
            if find(i) != root:
                connected = False
                break

        if connected:
            total = sum(w for _, _, w in combo)
            mod_total = total % K
            if mod_total < min_mod:
                min_mod = mod_total

    print(min_mod)

if __name__ == "__main__":
    main()