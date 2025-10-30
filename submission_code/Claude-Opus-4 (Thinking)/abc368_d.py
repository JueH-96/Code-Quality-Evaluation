from collections import defaultdict, deque

N, K = map(int, input().split())

adj = defaultdict(list)
for _ in range(N - 1):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

V = list(map(int, input().split()))

if K == 1:
    print(1)
else:
    # Find all vertices that must be included
    included = set()
    
    def find_path_bfs(start, end):
        if start == end:
            return [start]
        
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
                    
                    if v == end:
                        # Reconstruct path
                        path = []
                        curr = end
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        return path
        
        return []
    
    # Add all vertices on paths between pairs of V
    for i in range(K):
        for j in range(i + 1, K):
            path = find_path_bfs(V[i], V[j])
            included.update(path)
    
    print(len(included))