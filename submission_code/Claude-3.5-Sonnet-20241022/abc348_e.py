from collections import defaultdict, deque

def build_graph(N, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph

def bfs(start, graph, N):
    distances = [-1] * (N + 1)
    distances[start] = 0
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
                
    return distances

def solve():
    N = int(input())
    edges = []
    for _ in range(N-1):
        a, b = map(int, input().split())
        edges.append((a, b))
    
    C = list(map(int, input().split()))
    
    graph = build_graph(N, edges)
    
    min_sum = float('inf')
    for v in range(1, N+1):
        distances = bfs(v, graph, N)
        current_sum = 0
        for i in range(N):
            current_sum += C[i] * distances[i+1]
        min_sum = min(min_sum, current_sum)
    
    print(min_sum)

solve()