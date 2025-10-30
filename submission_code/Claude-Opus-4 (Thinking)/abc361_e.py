from collections import defaultdict, deque

def find_farthest(graph, start, n):
    dist = [-1] * (n + 1)
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        for v, w in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + w
                queue.append(v)
    
    max_dist = 0
    farthest_node = start
    for i in range(1, n + 1):
        if dist[i] > max_dist:
            max_dist = dist[i]
            farthest_node = i
    
    return farthest_node, max_dist

n = int(input())
graph = defaultdict(list)
total_weight = 0

for _ in range(n - 1):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))
    total_weight += c

# Find diameter using two BFS
node1, _ = find_farthest(graph, 1, n)
node2, diameter = find_farthest(graph, node1, n)

print(2 * total_weight - diameter)