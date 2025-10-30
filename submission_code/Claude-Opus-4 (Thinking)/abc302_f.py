from collections import deque, defaultdict

def solve():
    N, M = map(int, input().split())
    
    sets = []
    element_to_sets = defaultdict(list)
    
    for i in range(N):
        A = int(input())
        S = set(map(int, input().split()))
        sets.append(S)
        for elem in S:
            element_to_sets[elem].append(i)
    
    # Check if 1 and M exist
    if 1 not in element_to_sets or M not in element_to_sets:
        return -1
    
    # Check if any set contains both 1 and M
    for i in range(N):
        if 1 in sets[i] and M in sets[i]:
            return 0
    
    # Build adjacency list using inverted index
    adj = [set() for _ in range(N)]
    for elem, set_indices in element_to_sets.items():
        for i in range(len(set_indices)):
            for j in range(i + 1, len(set_indices)):
                u, v = set_indices[i], set_indices[j]
                adj[u].add(v)
                adj[v].add(u)
    
    # Find sets containing 1
    sets_with_1 = element_to_sets[1]
    
    # BFS from all sets containing 1
    dist = [-1] * N
    queue = deque()
    for start in sets_with_1:
        dist[start] = 0
        queue.append(start)
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    
    # Find minimum distance to any set containing M
    min_dist = float('inf')
    for end in element_to_sets[M]:
        if dist[end] != -1:
            min_dist = min(min_dist, dist[end])
    
    if min_dist == float('inf'):
        return -1
    else:
        return min_dist

print(solve())