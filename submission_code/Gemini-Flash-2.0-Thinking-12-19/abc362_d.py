import heapq

def solve():
    n, m = map(int, input().split())
    vertex_weights = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append(((u, v), b))
    
    adjacency_list = [[] for _ in range(n)]
    for (u, v), weight in edges:
        adjacency_list[u-1].append((v-1, weight))
        adjacency_list[v-1].append((u-1, weight))
        
    min_path_weights = [float('inf')] * n
    min_path_weights[0] = vertex_weights[0]
    
    priority_queue = [(min_path_weights[0], 0)]
    
    while priority_queue:
        current_weight, u = heapq.heappop(priority_queue)
        if current_weight > min_path_weights[u]:
            continue
            
        for v, edge_weight in adjacency_list[u]:
            new_weight = current_weight + edge_weight + vertex_weights[v]
            if new_weight < min_path_weights[v]:
                min_path_weights[v] = new_weight
                heapq.heappush(priority_queue, (new_weight, v))
                
    result = []
    for i in range(1, n):
        result.append(str(min_path_weights[i]))
        
    print(" ".join(result))

if __name__ == '__main__':
    solve()