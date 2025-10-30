def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    edges = data[2:]
    
    # Disjoint Set (Union-Find) initialization
    parent = list(range(N+1))
    comp_size = [1]*(N+1)         # size of connected component
    comp_edges = [0]*(N+1)        # number of edges in connected component

    def find(x):
        """Find with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        """Union by size and accumulate edge counts."""
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            # Merge smaller set into the bigger one
            if comp_size[root_a] < comp_size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            comp_size[root_a] += comp_size[root_b]
            comp_edges[root_a] += comp_edges[root_b] + 1
        else:
            # a, b already in the same set => just increase edge count
            comp_edges[root_a] += 1

    # Process the input edges
    idx = 0
    for _ in range(M):
        A = int(edges[idx])
        B = int(edges[idx+1])
        idx += 2
        union(A, B)

    # Ensure final path compression
    for i in range(1, N+1):
        find(i)

    # Calculate the answer:
    # For each connected component, we want (s*(s-1))//2 total edges;
    # we already have comp_edges[root]. The difference is the number of new friendships.
    answer = 0
    for i in range(1, N+1):
        if parent[i] == i:  # i is a root
            s = comp_size[i]
            e = comp_edges[i]
            answer += s*(s-1)//2 - e

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()