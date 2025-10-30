# YOUR CODE HERE
from collections import defaultdict, deque

def bfs_farthest(graph, start, n):
    dist = [-1] * (n + 1)
    dist[start] = 0
    queue = deque([start])
    
    farthest_node = start
    max_dist = 0
    
    while queue:
        u = queue.popleft()
        for v, w in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + w
                queue.append(v)
                if dist[v] > max_dist:
                    max_dist = dist[v]
                    farthest_node = v
    
    return farthest_node, max_dist

n = int(input())
graph = defaultdict(list)
total_weight = 0

for _ in range(n - 1):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))
    total_weight += c

# Find one end of the diameter
node1, _ = bfs_farthest(graph, 1, n)

# Find the other end of the diameter and the diameter length
node2, diameter = bfs_farthest(graph, node1, n)

# The answer is 2 * total_weight - diameter
print(2 * total_weight - diameter)