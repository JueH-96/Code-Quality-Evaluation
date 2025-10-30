def main():
    import sys
    from collections import defaultdict, deque

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [[] for _ in range(N + 1)]
    index = 1
    for _ in range(N - 1):
        u = int(data[index])
        v = int(data[index + 1])
        index += 2
        edges[u].append(v)
        edges[v].append(u)
    
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(edges[u])
    
    subgraph = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        if degree[u] not in [2, 3]:
            continue
        for v in edges[u]:
            if degree[v] in [2, 3]:
                subgraph[u].append(v)
    
    visited = [False] * (N + 1)
    total_pairs = 0
    
    for u in range(1, N + 1):
        if degree[u] in [2, 3] and not visited[u]:
            queue = deque([u])
            visited[u] = True
            c = 0
            if degree[u] == 2:
                c = 1
            while queue:
                current = queue.popleft()
                for neighbor in subgraph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        if degree[neighbor] == 2:
                            c += 1
            if c >= 2:
                total_pairs += c * (c - 1) // 2
    
    print(total_pairs)

if __name__ == "__main__":
    main()