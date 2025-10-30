import sys
import itertools
from collections import deque

def read_input():
    return map(int, sys.stdin.read().strip().split())

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def bfs_parity_count(graph, start_node):
    queue = deque([start_node])
    visited = [-1] * len(graph)
    visited[start_node] = 0
    parity_count = [0, 0]
    
    while queue:
        node = queue.popleft()
        current_parity = visited[node]
        parity_count[current_parity] += 1
        
        for neighbor in graph[node]:
            if visited[neighbor] == -1:
                visited[neighbor] = 1 - current_parity
                queue.append(neighbor)
            elif visited[neighbor] == current_parity:
                # This should not happen in a bipartite graph
                return False, (0, 0)
    
    return True, parity_count

def count_valid_graphs(N, P):
    max_edges = N * (N - 1) // 2
    min_edges = N - 1
    results = [0] * (max_edges - min_edges + 1)
    
    # Generate all possible edge sets
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for M in range(min_edges, max_edges + 1):
        for edges in itertools.combinations(all_edges, M):
            graph = [[] for _ in range(N)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            # Check if the graph is connected using BFS from node 0
            visited = [False] * N
            queue = deque([0])
            component_size = 0
            
            while queue:
                node = queue.popleft()
                if not visited[node]:
                    visited[node] = True
                    component_size += 1
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            queue.append(neighbor)
            
            if component_size != N:
                continue  # Not a connected graph
            
            # Check parity condition
            is_bipartite, parity_count = bfs_parity_count(graph, 0)
            if is_bipartite and parity_count[0] == parity_count[1]:
                results[M - min_edges] += 1
                results[M - min_edges] %= P
    
    return results

def main():
    N, P = read_input()
    assert 2 <= N <= 30 and N % 2 == 0
    assert 10**8 <= P <= 10**9 and is_prime(P)
    
    results = count_valid_graphs(N, P)
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()