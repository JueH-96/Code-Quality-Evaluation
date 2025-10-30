from collections import deque

def solve():
    N, M = map(int, input().split())
    
    sets = []
    contains_1 = []
    contains_M = []
    
    # Read all sets
    for i in range(N):
        A = int(input())
        S = list(map(int, input().split()))
        sets.append(set(S))
        
        if 1 in sets[i]:
            contains_1.append(i)
        if M in sets[i]:
            contains_M.append(i)
    
    # Check if any set already contains both 1 and M
    for i in range(N):
        if 1 in sets[i] and M in sets[i]:
            print(0)
            return
    
    # Build adjacency list - two sets are connected if they share an element
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if sets[i] & sets[j]:  # If intersection is not empty
                adj[i].append(j)
                adj[j].append(i)
    
    # BFS from all sets containing 1
    min_dist = float('inf')
    
    for start in contains_1:
        dist = [-1] * N
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            
            # Check if current set contains M
            if M in sets[u]:
                min_dist = min(min_dist, dist[u])
                break
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
    
    if min_dist == float('inf'):
        print(-1)
    else:
        print(min_dist)

solve()