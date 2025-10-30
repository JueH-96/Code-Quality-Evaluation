import sys
sys.setrecursionlimit(1 << 25)

def main():
    MOD = 998244353
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # 1-based indexing

    # Build children adjacency list
    children = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        u = A[v]
        children[u].append(v)

    # Find all cycles
    visited = [False] * (N + 1)
    in_cycle = [False] * (N + 1)
    cycles = []

    for i in range(1, N + 1):
        if not visited[i]:
            current = i
            path = []
            while True:
                if visited[current]:
                    if current in path:
                        idx = path.index(current)
                        cycle = path[idx:]
                        cycles.append(cycle)
                        # Mark all in cycle
                        for node in cycle:
                            in_cycle[node] = True
                            visited[node] = True
                    break
                visited[current] = True
                path.append(current)
                current = A[current]

    # Build processing order (post-order DFS for non-cycle nodes)
    visited = [False] * (N + 1)
    processing_order = []

    def dfs(u):
        visited[u] = True
        for v in children[u]:
            if not in_cycle[v] and not visited[v]:
                dfs(v)
        processing_order.append(u)

    for u in range(1, N + 1):
        if not in_cycle[u] and not visited[u]:
            dfs(u)

    # Initialize DP table
    F = [[0] * (M + 2) for _ in range(N + 2)]  # F[u][a] for a in 1..M

    for u in processing_order:
        if not children[u]:
            # Leaf node: F[u][a] = a
            for a in range(1, M + 1):
                F[u][a] = a % MOD
        else:
            # Compute product of children's F at each x
            G = [1] * (M + 2)  # G[x] for x in 1..M
            for c in children[u]:
                for x in range(1, M + 1):
                    G[x] = (G[x] * F[c][x]) % MOD
            # Compute prefix sums
            prefix = 0
            for x in range(1, M + 1):
                prefix = (prefix + G[x]) % MOD
                F[u][x] = prefix

    # Process each cycle component
    total = 1
    for cycle in cycles:
        # Collect all roots (direct children of cycle nodes not in cycle)
        roots = []
        for u in cycle:
            for v in children[u]:
                if not in_cycle[v]:
                    roots.append(v)
        # Compute contribution for this component
        contrib = 0
        for v in range(1, M + 1):
            prod = 1
            for r in roots:
                prod = (prod * F[r][v]) % MOD
            contrib = (contrib + prod) % MOD
        total = (total * contrib) % MOD

    print(total % MOD)

if __name__ == "__main__":
    main()