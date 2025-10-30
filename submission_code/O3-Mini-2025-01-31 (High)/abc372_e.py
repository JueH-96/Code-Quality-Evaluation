def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Read number of vertices and queries.
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    # DSU initialization: parent[i] is the representative of node i.
    # size[i] holds the size of the component (used for union by size).
    # comp_best[i] holds the list of at most 10 largest vertices (sorted descending)
    # in the component whose representative is i.
    parent = list(range(n + 1))
    size = [1] * (n + 1)
    comp_best = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        comp_best[i] = [i]
    
    # Find function with path compression (iterative implementation).
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Merge two sorted (in descending order) lists, keeping only the top 10 elements.
    def merge_lists(list1, list2):
        i = j = 0
        merged = []
        while len(merged) < 10 and i < len(list1) and j < len(list2):
            if list1[i] >= list2[j]:
                merged.append(list1[i])
                i += 1
            else:
                merged.append(list2[j])
                j += 1
        while len(merged) < 10 and i < len(list1):
            merged.append(list1[i])
            i += 1
        while len(merged) < 10 and j < len(list2):
            merged.append(list2[j])
            j += 1
        return merged

    # Union function: merges the components of a and b.
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        comp_best[ra] = merge_lists(comp_best[ra], comp_best[rb])
    
    # Process queries
    out_lines = []
    for _ in range(q):
        typ = int(next(it))
        if typ == 1:
            u = int(next(it))
            v = int(next(it))
            union(u, v)
        else:
            v = int(next(it))
            k = int(next(it))
            rep = find(v)
            if k > len(comp_best[rep]):
                out_lines.append("-1")
            else:
                out_lines.append(str(comp_best[rep][k - 1]))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()