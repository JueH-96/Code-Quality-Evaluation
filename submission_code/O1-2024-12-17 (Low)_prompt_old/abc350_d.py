def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = input_data[2:]
    
    # Disjoint Set Union (Union-Find) implementation
    parent = list(range(N))
    size = [1]*N
    comp_edges = [0]*N  # count edges within each component
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA != rootB:
            if size[rootA] < size[rootB]:
                rootA, rootB = rootB, rootA
            parent[rootB] = rootA
            size[rootA] += size[rootB]
            comp_edges[rootA] += comp_edges[rootB]
            comp_edges[rootB] = 0
    
    # Read edges, union
    idx = 0
    for _ in range(M):
        a = int(edges[idx]) - 1
        b = int(edges[idx+1]) - 1
        idx += 2
        # union their sets
        ra = find(a)
        rb = find(b)
        comp_edges[ra] += 1  # add edge count to the root
        if ra != rb:
            comp_edges[rb] += 0  # ensure we have a slot
            union(a, b)
    
    # Now calculate the sum of new edges that can be formed
    # Each connected component can become a complete graph
    # If a component has c nodes, it can have c*(c-1)//2 edges in total
    # The number of new edges is sum of [c*(c-1)//2 - current_edges_in_that_component]
    answer = 0
    seen = set()
    for i in range(N):
        r = find(i)
        if r not in seen:
            seen.add(r)
            c = size[r]
            e = comp_edges[r]
            answer += c*(c-1)//2 - e
    
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()