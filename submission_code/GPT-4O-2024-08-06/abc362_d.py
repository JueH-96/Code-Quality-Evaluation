import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index + N]))
    index += N
    
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        U = int(data[index])
        index += 1
        V = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        graph[U].append((V, B))
        graph[V].append((U, B))
    
    # Dijkstra's algorithm
    min_weight = [float('inf')] * (N + 1)
    min_weight[1] = A[0]  # 1-based index, A[0] is the weight of vertex 1
    
    pq = [(A[0], 1)]  # (current path weight, vertex)
    
    while pq:
        current_weight, u = heapq.heappop(pq)
        
        if current_weight > min_weight[u]:
            continue
        
        for v, edge_weight in graph[u]:
            new_weight = current_weight + edge_weight + A[v - 1]
            if new_weight < min_weight[v]:
                min_weight[v] = new_weight
                heapq.heappush(pq, (new_weight, v))
    
    # Output the results for vertices 2 to N
    result = [str(min_weight[i]) for i in range(2, N + 1)]
    print(" ".join(result))