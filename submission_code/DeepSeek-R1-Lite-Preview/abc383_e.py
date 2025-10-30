def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict

    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u-1, v-1))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    A = [a-1 for a in A]
    B = [b-1 for b in B]

    # Build MST using Krusky's algorithm
    parent = [i for i in range(N)]
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    edges.sort()
    mst = []
    for w, u, v in edges:
        pu = find(u)
        pv = find(v)
        if pu != pv:
            mst.append((w, u, v))
            parent[pu] = pv
    # Build adjacency list for MST
    adj = defaultdict(list)
    for w, u, v in mst:
        adj[u].append((v, w))
        adj[v].append((u, w))
    # DFS to compute count_A and count_B for each subtree
    count_A = [0] * N
    count_B = [0] * N
    # Precompute count_A and count_B for each node
    for a in A:
        count_A[a] += 1
    for b in B:
        count_B[b] += 1
    total_sum = 0
    def dfs(u, parent_node):
        nonlocal total_sum
        for v, w in adj[u]:
            if v != parent_node:
                dfs(v, u)
                count_A[u] += count_A[v]
                count_B[u] += count_B[v]
                surplus_B = count_B[v] - count_A[v]
                if surplus_B > 0:
                    total_sum += surplus_B * w
    # Root the tree at node 0
    dfs(0, -1)
    print(total_sum)

if __name__ == '__main__':
    main()