from collections import defaultdict, deque

n = int(input())

graph = defaultdict(list)
total_weight = 0

for _ in range(n - 1):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))
    total_weight += c

def bfs_farthest(start):
    visited = [False] * (n + 1)
    queue = deque([(start, 0)])
    visited[start] = True
    farthest_node = start
    max_dist = 0
    
    while queue:
        node, dist = queue.popleft()
        if dist > max_dist:
            max_dist = dist
            farthest_node = node
        
        for neighbor, weight in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, dist + weight))
    
    return farthest_node, max_dist

# Find one end of diameter
end1, _ = bfs_farthest(1)

# Find the other end and the diameter length
end2, diameter = bfs_farthest(end1)

# Answer is 2 * total_weight - diameter
answer = 2 * total_weight - diameter
print(answer)