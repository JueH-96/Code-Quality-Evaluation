def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N - 1):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        edges.append((u, v))
        index += 2
    
    from collections import defaultdict, deque
    
    # Create adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Function to find path using BFS
    def find_path(start, end):
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == end:
                break
            for neighbor in adj[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        # Reconstruct the path
        path = []
        step = end
        while step is not None:
            path.append(step)
            step = parent[step]
        path.reverse()
        return path
    
    # Count valid pairs
    count = 0
    for u in range(N):
        for v in range(u + 1, N):
            if v not in adj[u]:
                # Find path from u to v
                path = find_path(u, v)
                # Check if all nodes in the path can have degree 3
                if all(len(adj[node]) == 2 for node in path):
                    count += 1
    
    print(count)