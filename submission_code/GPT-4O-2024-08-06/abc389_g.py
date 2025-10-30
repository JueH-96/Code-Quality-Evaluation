# YOUR CODE HERE
import sys
from itertools import combinations
from collections import deque

def is_connected(graph, N):
    visited = [False] * N
    queue = deque([0])
    visited[0] = True
    count = 1
    
    while queue:
        node = queue.popleft()
        for neighbor in range(N):
            if graph[node][neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1
    
    return count == N

def even_odd_distance_check(graph, N):
    distances = [-1] * N
    queue = deque([0])
    distances[0] = 0
    
    while queue:
        node = queue.popleft()
        for neighbor in range(N):
            if graph[node][neighbor] and distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    even_count = sum(1 for d in distances if d % 2 == 0)
    odd_count = N - even_count
    return even_count == odd_count

def count_graphs(N, P):
    max_edges = N * (N - 1) // 2
    results = []
    
    for M in range(N-1, max_edges + 1):
        count = 0
        edges = list(combinations(range(N), 2))
        
        for edge_combination in combinations(edges, M):
            graph = [[0] * N for _ in range(N)]
            for u, v in edge_combination:
                graph[u][v] = 1
                graph[v][u] = 1
            
            if is_connected(graph, N) and even_odd_distance_check(graph, N):
                count += 1
        
        results.append(count % P)
    
    return results

def main():
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    P = int(data[1])
    
    results = count_graphs(N, P)
    print(" ".join(map(str, results)))