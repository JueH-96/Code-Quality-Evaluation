def min_operations_to_delete_vertex_1():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N - 1):
        u = int(data[index])
        v = int(data[index + 1])
        edges.append((u, v))
        index += 2
    
    from collections import deque, defaultdict
    
    # Build the adjacency list for the tree
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)
    
    # BFS to find the maximum depth from node 1
    queue = deque([1])
    visited = [False] * (N + 1)
    visited[1] = True
    depth = 0
    
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            for neighbor in tree[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    
    # The maximum depth found is the minimum number of operations needed
    print(depth - 1)