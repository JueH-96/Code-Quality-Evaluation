# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    index = 1
    for _ in range(N - 1):
        U = int(data[index]) - 1
        V = int(data[index + 1]) - 1
        L = int(data[index + 2])
        edges.append((U, V, L))
        index += 3
    
    # Build the adjacency list for the tree
    tree = defaultdict(list)
    for U, V, L in edges:
        tree[U].append((V, L))
        tree[V].append((U, L))
    
    def bfs(start):
        dist = [-1] * N
        dist[start] = 0
        queue = deque([start])
        farthest_node = start
        max_distance = 0
        
        while queue:
            node = queue.popleft()
            current_distance = dist[node]
            
            for neighbor, length in tree[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = current_distance + length
                    queue.append(neighbor)
                    if dist[neighbor] > max_distance:
                        max_distance = dist[neighbor]
                        farthest_node = neighbor
        
        return farthest_node, max_distance
    
    # Find the farthest node from node 0
    farthest_from_1, dist_from_1 = bfs(0)
    
    # Find the diameter of the tree
    farthest_from_A, diameter = bfs(farthest_from_1)
    
    # Output the results
    results = [0] * N
    results[0] = 2 * dist_from_1
    if N > 1:
        for k in range(1, N):
            results[k] = 2 * diameter
    
    for result in results:
        print(result)