def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    # Read N and M
    N, M = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]

    # DSU (Disjoint Set Union) for the graph
    parent = list(range(N))
    size = [1] * N

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    # Build the DSU structure from the edges
    for u, v in edges:
        # Convert to 0-based
        union(u - 1, v - 1)

    # Read K (forbidden pairs)
    K = int(input())
    forbidden = {}  # forbidden[a] = set of components b that must not merge with a

    # We'll precompute each vertex's component
    for _ in range(K):
        x, y = map(int, input().split())
        # Convert to 0-based
        rx, ry = find(x - 1), find(y - 1)
        # Given G is good, rx != ry is guaranteed
        if rx > ry:
            rx, ry = ry, rx
        if rx not in forbidden:
            forbidden[rx] = set()
        forbidden[rx].add(ry)

    # Precompute the component ID for each vertex
    comp = [find(i) for i in range(N)]

    # Read Q, and process each query
    Q = int(input())
    results = []
    for _ in range(Q):
        p, q = map(int, input().split())
        # Convert to 0-based
        cp, cq = comp[p - 1], comp[q - 1]
        if cp == cq:
            # Adding this edge doesn't connect any previously unconnected pairs
            results.append("Yes")
        else:
            # Check if this merges two components that contain a forbidden pair
            if cp > cq:
                cp, cq = cq, cp
            if cp in forbidden and cq in forbidden[cp]:
                results.append("No")
            else:
                results.append("Yes")

    print("
".join(results))

# Do not forget to call main.
main()