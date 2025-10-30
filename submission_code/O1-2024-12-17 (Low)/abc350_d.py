def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges_list = input_data[2:]
    
    # Disjoint Set Union (Union-Find) implementation
    parent = list(range(N))
    rank = [0]*N
    size = [1]*N  # track component sizes
    # We'll also keep track of the number of edges in each component
    edge_count = [0]*N

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA != rootB:
            if rank[rootA] < rank[rootB]:
                rootA, rootB = rootB, rootA
            parent[rootB] = rootA
            size[rootA] += size[rootB]
            edge_count[rootA] += edge_count[rootB]
            if rank[rootA] == rank[rootB]:
                rank[rootA] += 1

    idx = 0
    for _ in range(M):
        A = int(edges_list[idx]) - 1
        B = int(edges_list[idx+1]) - 1
        idx += 2
        # Union the sets
        union(A, B)
        # After union, find the new root and increment its edge count
        r = find(A)
        edge_count[r] += 1

    # Now compute the sum of (k*(k-1)/2 - e) over all connected components
    # We'll identify each representative once and sum accordingly
    seen = set()
    answer = 0
    for i in range(N):
        r = find(i)
        if r not in seen:
            seen.add(r)
            k = size[r]
            e = edge_count[r]
            # number of edges that can still be added to form a complete subgraph
            answer += (k*(k-1)//2) - e

    print(answer)

# Don't forget to call main()
if __name__ == "__main__":
    main()