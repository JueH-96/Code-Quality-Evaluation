import heapq

def solve():
    n, m = map(int, input().split())
    vertex_weights = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append(((u, v), b))
    
    adjacency_list = [[] for _ in range(n + 1)]
    for (u, v), weight in edges:
        adjacency_list[u].append((v, weight))
        adjacency_list[v].append((u, weight))
        
    distances = [float('inf')] * (n + 1)
    distances[1] = vertex_weights[0]
    
    priority_queue = [(distances[1], 1)]
    
    while priority_queue:
        current_weight, u = heapq.heappop(priority_queue)
        if current_weight > distances[u]:
            continue
            
        for v, edge_weight in adjacency_list[u]:
            new_weight = distances[u] + edge_weight + vertex_weights[v-1]
            if new_weight < distances[v]:
                distances[v] = new_weight
                heapq.heappush(priority_queue, (distances[v], v))
                
    result = []
    for i in range(2, n + 1):
        result.append(str(distances[i]))
        
    print(" ".join(result))

if __name__ == '__main__':
    solve()