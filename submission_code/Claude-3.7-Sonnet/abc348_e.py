from collections import defaultdict, deque

def solve():
    # Read input
    N = int(input())
    tree = defaultdict(list)
    
    for _ in range(N - 1):
        a, b = map(int, input().split())
        tree[a].append(b)
        tree[b].append(a)
    
    C = list(map(int, input().split()))
    
    # Choose vertex 1 as the root
    root = 1
    
    # Calculate distances from root to all vertices using BFS
    distances = [0] * (N + 1)
    visited = [False] * (N + 1)
    queue = deque([(root, 0)])
    visited[root] = True
    
    while queue:
        node, d = queue.popleft()
        distances[node] = d
        
        for neighbor in tree[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, d + 1))
    
    # Calculate f(root)
    f_root = sum(C[i - 1] * distances[i] for i in range(1, N + 1))
    
    # Construct a rooted tree
    parent = [-1] * (N + 1)
    children = [[] for _ in range(N + 1)]
    visited = [False] * (N + 1)
    
    # BFS to populate parent and children
    queue = deque([root])
    visited[root] = True
    
    while queue:
        node = queue.popleft()
        
        for neighbor in tree[node]:
            if not visited[neighbor]:
                parent[neighbor] = node
                children[node].append(neighbor)
                visited[neighbor] = True
                queue.append(neighbor)
    
    # Calculate subtree weights using an iterative post-order traversal
    subtree_weights = [0] * (N + 1)
    
    # Iterative post-order traversal
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        
        if processed:
            subtree_weights[node] = C[node - 1]
            for child in children[node]:
                subtree_weights[node] += subtree_weights[child]
        else:
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))
    
    total_weight = sum(C)
    
    # Calculate f(v) for all vertices
    f_values = [0] * (N + 1)
    f_values[root] = f_root
    
    # BFS to calculate f(v) for all vertices
    queue = deque([root])
    visited = [False] * (N + 1)
    visited[root] = True
    
    while queue:
        node = queue.popleft()
        
        for child in children[node]:
            f_values[child] = f_values[node] + total_weight - 2 * subtree_weights[child]
            visited[child] = True
            queue.append(child)
    
    # Find the minimum f value
    min_f = min(f_values[1:N+1])
    
    return min_f

print(solve())