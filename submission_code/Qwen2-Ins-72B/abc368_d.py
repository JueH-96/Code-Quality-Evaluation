from collections import defaultdict, deque

def dfs(graph, root, required):
    visited = set()
    queue = deque([(root, 0)])
    parent = {}
    depth = {}
    while queue:
        node, d = queue.popleft()
        visited.add(node)
        depth[node] = d
        for neighbor in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                queue.append((neighbor, d + 1))
    subtree = set(required)
    for node in required:
        while node != root:
            subtree.add(parent[node])
            node = parent[node]
    return subtree, depth

def main():
    N, K = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(N - 1):
        A, B = map(int, input().split())
        graph[A].append(B)
        graph[B].append(A)
    required = set(map(int, input().split()))
    subtree, depth = dfs(graph, 1, required)
    root = min(subtree, key=lambda x: depth[x])
    _, depth = dfs(graph, root, required)
    print(len({depth[node] for node in required}))

if __name__ == "__main__":
    main()