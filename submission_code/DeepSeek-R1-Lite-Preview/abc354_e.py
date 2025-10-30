def main():
    import sys
    from itertools import combinations
    from collections import deque

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        cards.append((A, B))
        index += 2

    # Build graph
    adj = [[] for _ in range(N)]
    A_groups = {}
    B_groups = {}

    for idx, (A, B) in enumerate(cards):
        if A in A_groups:
            A_groups[A].append(idx)
        else:
            A_groups[A] = [idx]
        if B in B_groups:
            B_groups[B].append(idx)
        else:
            B_groups[B] = [idx]

    for group in A_groups.values():
        for u, v in combinations(group, 2):
            adj[u].append(v)
            adj[v].append(u)
    for group in B_groups.values():
        for u, v in combinations(group, 2):
            adj[u].append(v)
            adj[v].append(u)

    # Find connected components
    visited = [False] * N
    components = []
    for i in range(N):
        if not visited[i]:
            queue = deque([i])
            visited[i] = True
            component = []
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)

    # Function to compute Grundy number for a component
    def compute_grundy(component, adj):
        n = len(component)
        # Map component indices to 0 to n-1
        index_map = {node: idx for idx, node in enumerate(component)}
        # Build adjacency for the component
        comp_adj = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[component[u]]:
                if v in index_map:
                    comp_adj[u].append(v)
        # Precompute edges
        edges = []
        for u in range(n):
            for v in comp_adj[u]:
                if u < index_map[v]:
                    edges.append((u, index_map[v]))
        # DP: grundy[mask] where mask is subset of nodes
        grundy = [0] * (1 << n)
        for mask in range(1, 1 << n):
            moves = set()
            for u, v in edges:
                if (mask & (1 << u)) and (mask & (1 << v)):
                    next_mask = mask & (~((1 << u) | (1 << v)))
                    moves.add(grundy[next_mask])
            mex = 0
            while mex in moves:
                mex += 1
            grundy[mask] = mex
        return grundy[(1 << n) - 1]

    # Compute Grundy numbers for each component
    overall_grundy = 0
    for comp in components:
        if len(comp) == 1:
            # Single card cannot be removed; Grundy number is 0
            grundy = 0
        else:
            grundy = compute_grundy(comp, adj)
        overall_grundy ^= grundy

    # Determine the winner
    if overall_grundy != 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()