from collections import deque

def solve():
    n, m = map(int, input().split())
    sets = []
    
    for _ in range(n):
        a = int(input())
        s = set(map(int, input().split()))
        sets.append(s)
    
    # Check if any set already contains both 1 and M
    for s in sets:
        if 1 in s and m in s:
            return 0
    
    # Build a graph where nodes are sets and edges connect sets with common elements
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if sets[i].intersection(sets[j]):  # if sets have common elements
                graph[i].append(j)
                graph[j].append(i)
    
    # Find which sets contain 1 and which contain M
    sets_with_1 = [i for i, s in enumerate(sets) if 1 in s]
    sets_with_m = [i for i, s in enumerate(sets) if m in s]
    
    if not sets_with_1 or not sets_with_m:
        return -1
    
    # BFS to find the shortest path from any set containing 1 to any set containing M
    visited = [False] * n
    queue = deque()
    distance = [float('inf')] * n
    
    for i in sets_with_1:
        queue.append(i)
        visited[i] = True
        distance[i] = 0
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    
    # Check if we can reach any set containing M
    min_dist = float('inf')
    for i in sets_with_m:
        min_dist = min(min_dist, distance[i])
    
    if min_dist == float('inf'):
        return -1
    else:
        return min_dist

print(solve())