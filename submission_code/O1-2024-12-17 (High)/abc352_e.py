def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    subsets = []
    for _ in range(M):
        K_i, C_i = map(int, input().split())
        S_i = list(map(int, input().split()))
        subsets.append((C_i, S_i))

    # Sort the subsets by their weight
    subsets.sort(key=lambda x: x[0])

    # Disjoint Set Union (Union-Find) for 1-based indexing
    parent = list(range(N+1))
    size = [1]*(N+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a = find(a)
        b = find(b)
        if a == b:
            return False
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        return True

    mst_cost = 0
    used_edges = 0

    # Kruskal-like approach: for each subset, attempt to connect
    # all vertices in that subset via a "star" to avoid enumerating all edges.
    for w, S_i in subsets:
        pivot = S_i[0]
        for v in S_i[1:]:
            if find(pivot) != find(v):
                union(pivot, v)
                mst_cost += w
                used_edges += 1
                if used_edges == N - 1:
                    break
        if used_edges == N - 1:
            break

    # If we used fewer than N-1 edges, the graph is not connected
    if used_edges < N - 1:
        print(-1)
    else:
        print(mst_cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()