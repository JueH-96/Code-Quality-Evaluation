from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    edges = data[1:n+1]

    # Build undirected adjacency list
    undirected = defaultdict(list)
    for s in edges:
        u = s[0]
        v = s[1]
        undirected[u].append(v)
        undirected[v].append(u)

    # Compute in_degree and out_degree
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for s in edges:
        u = s[0]
        v = s[1]
        out_degree[u] += 1
        in_degree[v] += 1

    # Find connected components
    visited = set()
    components = []
    for u in undirected:
        if u not in visited:
            queue = deque()
            queue.append(u)
            visited.add(u)
            component = set()
            component.add(u)
            while queue:
                current = queue.popleft()
                for neighbor in undirected[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        component.add(neighbor)
            components.append(component)

    total = 0

    for comp in components:
        has_edges = any((out_degree.get(u, 0) > 0) for u in comp)
        if has_edges:
            sources = 0
            sinks = 0
            for u in comp:
                if out_degree.get(u, 0) > in_degree.get(u, 0):
                    sources += 1
                if in_degree.get(u, 0) > out_degree.get(u, 0):
                    sinks += 1
            if sources == 0 and sinks == 0:
                total += 1
            else:
                total += max(sources, sinks)
        else:
            total += 1

    print(total)

if __name__ == '__main__':
    main()