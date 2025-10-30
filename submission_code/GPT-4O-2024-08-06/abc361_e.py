# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    total_length = 0
    
    index = 1
    for _ in range(N - 1):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        C = int(data[index + 2])
        edges.append((A, B, C))
        total_length += C
        index += 3
    
    # Build the graph
    graph = defaultdict(list)
    for A, B, C in edges:
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    # Function to find the longest path from a given start node
    def longest_path_from(start):
        # BFS to find the farthest node from start
        queue = deque([(start, 0)])
        visited = [False] * N
        visited[start] = True
        farthest_node = start
        max_distance = 0
        
        while queue:
            node, dist = queue.popleft()
            if dist > max_distance:
                max_distance = dist
                farthest_node = node
            for neighbor, length in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + length))
        
        return max_distance
    
    # Start from any node, say node 0
    longest_path = longest_path_from(0)
    
    # The minimum travel distance is twice the total length minus the longest path
    min_travel_distance = 2 * total_length - longest_path
    print(min_travel_distance)