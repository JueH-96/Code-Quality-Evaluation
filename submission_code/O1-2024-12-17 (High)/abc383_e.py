def main():
    import sys
    input = sys.stdin.readline
    sys.setrecursionlimit(10**7)

    # Read basic inputs
    N, M, K = map(int, input().split())
    
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((w, u, v))

    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Count how many times each vertex appears in A or B
    # (A_i != B_j guarantees no vertex is in both A and B.)
    cA = [0]*(N+1)
    cB = [0]*(N+1)
    for a in A:
        cA[a] += 1
    for b in B:
        cB[b] += 1

    # Disjoint Set Union (Union-Find) utilities
    parent = list(range(N+1))
    rank = [0]*(N+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx, 0
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        
        # Calculate how many new cross-matches can form
        # between unmatched A's in one component and unmatched B's in the other.
        cross1 = min(cA[rx], cB[ry])  # A in rx with B in ry
        cross2 = min(cA[ry], cB[rx])  # A in ry with B in rx
        cross = cross1 + cross2

        # Update the counts of unmatched A/B in the new root
        cA[rx] = cA[rx] + cA[ry] - cross
        cB[rx] = cB[rx] + cB[ry] - cross

        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

        return rx, cross

    # Sort edges by ascending weight
    edges.sort(key=lambda e: e[0])

    matched = 0
    total_cost = 0

    # Kruskal-like process
    for w, u, v in edges:
        root, new_pairs = union(u, v)
        if new_pairs > 0:
            total_cost += w * new_pairs
            matched += new_pairs
            if matched == K:
                break

    print(total_cost)

# Do not forget to call main at the end!
if __name__ == "__main__":
    main()