def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # DSU (Union-Find)
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    # Process the edges
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        union(u, v)
    
    # Count connected components
    components = set()
    for i in range(n):
        components.add(find(i))
    comp_count = len(components)
    
    # For a forest, we want to keep (n - comp_count) edges.
    # Therefore, minimum number of deletions: m - (n - comp_count)
    answer = m - (n - comp_count)
    
    sys.stdout.write(str(answer))

if __name__ == "__main__":
    main()