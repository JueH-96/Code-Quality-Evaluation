# YOUR CODE HERE
n = int(input())
graph = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

count = 0
deleted = [False] * (n + 1)
while not deleted[1]:
    leaves = []
    for i in range(1, n + 1):
        if not deleted[i] and len(graph[i]) <= 1:
            leaves.append(i)
    
    if 1 in leaves:
        deleted[1] = True
        count +=1
        break
    
    farthest_leaf = -1
    max_dist = -1
    q = [1]
    visited = [False]*(n+1)
    visited[1] = True
    dist = [-1]*(n+1)
    dist[1] = 0
    
    while q:
        curr = q.pop(0)
        for neighbor in graph[curr]:
            if not visited[neighbor] and not deleted[neighbor]:
                visited[neighbor] = True
                dist[neighbor] = dist[curr] + 1
                q.append(neighbor)
                if not deleted[neighbor] and len(graph[neighbor]) <=1:
                    if dist[neighbor] > max_dist:
                        max_dist = dist[neighbor]
                        farthest_leaf = neighbor
    
    
    deleted[farthest_leaf] = True
    count += 1
    for i in range(1,n+1):
        if farthest_leaf in graph[i]:
            graph[i].remove(farthest_leaf)

print(count)