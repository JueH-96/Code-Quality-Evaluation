from collections import defaultdict, deque

N, Q = map(int, input().split())
graph = defaultdict(list)

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Add edge
        u, v = query[1], query[2]
        graph[u].append(v)
        graph[v].append(u)
    
    else:  # Type 2 query
        v, k = query[1], query[2]
        
        # Find all connected vertices using BFS
        visited = set()
        queue = deque([v])
        visited.add(v)
        connected = []
        
        while queue:
            curr = queue.popleft()
            connected.append(curr)
            
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Sort in descending order
        connected.sort(reverse=True)
        
        # Get k-th largest
        if k <= len(connected):
            print(connected[k-1])
        else:
            print(-1)