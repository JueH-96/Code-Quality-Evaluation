def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    
    # DSU arrays
    parent = list(range(n + 1))
    size = [1] * (n + 1)
    # For each component representative, store up to 10 largest elements (in descending order)
    comp_vals = [[i] for i in range(n + 1)]
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # Union by size, attach smaller to larger
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        # Now attach rb to ra
        parent[rb] = ra
        size[ra] += size[rb]
        # Merge comp_vals[ra] and comp_vals[rb]
        merged = []
        i = 0
        j = 0
        list1 = comp_vals[ra]
        list2 = comp_vals[rb]
        # Both lists are in descending order
        while i < len(list1) and j < len(list2) and len(merged) < 10:
            if list1[i] >= list2[j]:
                merged.append(list1[i])
                i += 1
            else:
                merged.append(list2[j])
                j += 1
        while i < len(list1) and len(merged) < 10:
            merged.append(list1[i])
            i += 1
        while j < len(list2) and len(merged) < 10:
            merged.append(list2[j])
            j += 1
        comp_vals[ra] = merged
    
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
            if k <= len(comp_vals[rep]):
                # k-th largest:
                out_lines.append(str(comp_vals[rep][k - 1]))
            else:
                out_lines.append(str(-1))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()