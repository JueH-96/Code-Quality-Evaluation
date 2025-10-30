from collections import deque, defaultdict

n = int(input())

# Build adjacency list
graph = defaultdict(list)
for _ in range(n - 1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

c = list(map(int, input().split()))

def bfs_distances(start):
    distances = [-1] * (n + 1)
    distances[start] = 0
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances

min_f = float('inf')

for v in range(1, n + 1):
    distances = bfs_distances(v)
    f_v = 0
    for i in range(1, n + 1):
        f_v += c[i - 1] * distances[i]
    min_f = min(min_f, f_v)

print(min_f)