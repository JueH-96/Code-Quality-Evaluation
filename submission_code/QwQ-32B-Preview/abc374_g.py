def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    product_names = data[1:N+1]

    # Create a graph where keys are letters and values are sets of letters forming valid product names
    graph = defaultdict(set)
    for pn in product_names:
        a, b = pn[0], pn[1]
        graph[a].add(b)
        graph[b].add(a)

    # Set of used product names
    used = set(product_names)

    def dfs(node, path):
        # Visit each neighbor of the current node
        for neighbor in list(graph[node]):
            # Form the product name (considering both orders)
            if node < neighbor:
                pn = node + neighbor
            else:
                pn = neighbor + node
            # If the product name has not been used, use it and continue DFS
            if pn in used:
                used.remove(pn)
                dfs(neighbor, path + [pn])
                # No need to backtrack as we are not reusing edges

    count = 0
    while used:
        # Start a new path with any remaining used product name
        start_pn = next(iter(used))
        a, b = start_pn[0], start_pn[1]
        used.remove(start_pn)
        dfs(a, [start_pn])
        count += 1

    print(count)

if __name__ == "__main__":
    main()