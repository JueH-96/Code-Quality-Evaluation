from collections import deque, defaultdict

def solve():
    N, M = map(int, input().split())
    
    sets = []
    for i in range(N):
        line = list(map(int, input().split()))
        A = line[0]
        elements = set(line[1:])
        sets.append(elements)
    
    # Check if any set already contains both 1 and M
    for s in sets:
        if 1 in s and M in s:
            return 0
    
    # Build adjacency list - two sets are connected if they share at least one element
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if sets[i] & sets[j]:  # If intersection is non-empty
                adj[i].append(j)
                adj[j].append(i)
    
    # Find sets containing 1 and sets containing M
    start_sets = []
    end_sets = []
    
    for i in range(N):
        if 1 in sets[i]:
            start_sets.append(i)
        if M in sets[i]:
            end_sets.append(i)
    
    if not start_sets or not end_sets:
        return -1
    
    # BFS from all sets containing 1
    queue = deque()
    visited = [False] * N
    
    for start in start_sets:
        queue.append((start, 0))
        visited[start] = True
    
    while queue:
        node, dist = queue.popleft()
        
        # Check if current set contains M
        if M in sets[node]:
            return dist
        
        # Explore neighbors
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))
    
    return -1

print(solve())