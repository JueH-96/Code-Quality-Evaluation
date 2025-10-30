import heapq
import sys
from itertools import permutations

def dijkstra_max_edge(graph, start, n):
    max_edge = [float('inf')] * (n + 1)
    max_edge[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_max_edge, u = heapq.heappop(pq)
        
        if current_max_edge > max_edge[u]:
            continue
        
        for weight, v in graph[u]:
            new_max_edge = max(current_max_edge, weight)
            if new_max_edge < max_edge[v]:
                max_edge[v] = new_max_edge
                heapq.heappush(pq, (new_max_edge, v))
    
    return max_edge

def main():
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    M = int(data[idx + 1])
    K = int(data[idx + 2])
    idx += 3
    
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx + 1])
        w = int(data[idx + 2])
        graph[u].append((w, v))
        graph[v].append((w, u))
        idx += 3
    
    A = list(map(int, data[idx:idx + K]))
    idx += K
    B = list(map(int, data[idx:idx + K]))
    
    # Precompute all-pairs minimum path weights using modified Dijkstra
    all_pairs_max_edge = {}
    for i in range(1, N + 1):
        all_pairs_max_edge[i] = dijkstra_max_edge(graph, i, N)
    
    # Use permutations to find the optimal assignment
    min_sum = float('inf')
    
    for perm in permutations(B):
        current_sum = 0
        for i in range(K):
            current_sum += all_pairs_max_edge[A[i]][perm[i]]
        min_sum = min(min_sum, current_sum)
    
    print(min_sum)

if __name__ == "__main__":
    main()