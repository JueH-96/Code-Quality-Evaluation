from collections import deque

N, M = map(int, input().split())
sets = []
for i in range(N):
    line = list(map(int, input().split()))
    A = line[0]
    s = set(line[1:])
    sets.append(s)

# Check if any set already contains both 1 and M
found = False
for s in sets:
    if 1 in s and M in s:
        print(0)
        found = True
        break

if not found:
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if sets[i] & sets[j]:  # They share at least one element
                adj[i].append(j)
                adj[j].append(i)
    
    # Find sets containing 1 and M
    start_sets = []
    end_sets = []
    for i in range(N):
        if 1 in sets[i]:
            start_sets.append(i)
        if M in sets[i]:
            end_sets.append(i)
    
    # BFS from start_sets to end_sets
    queue = deque()
    visited = [False] * N
    
    for start in start_sets:
        queue.append((start, 0))
        visited[start] = True
    
    result = -1
    while queue:
        node, dist = queue.popleft()
        
        if node in end_sets:
            result = dist
            break
        
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + 1))
    
    print(result)