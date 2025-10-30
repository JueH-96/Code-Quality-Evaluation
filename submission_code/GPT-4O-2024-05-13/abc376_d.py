# YOUR CODE HERE
import sys
from collections import deque, defaultdict

def find_min_cycle_with_vertex_one(N, M, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    
    # BFS to find the shortest cycle involving vertex 1
    def bfs(start):
        queue = deque([(start, 0)])  # (current_node, distance)
        visited = {start: 0}
        parent = {start: None}
        
        while queue:
            current, dist = queue.popleft()
            
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    parent[neighbor] = current
                    queue.append((neighbor, dist + 1))
                elif neighbor == start:
                    # Found a cycle
                    return dist + 1
        
        return -1
    
    min_cycle_length = bfs(1)
    print(min_cycle_length)

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        edges.append((a, b))
        index += 2
    
    find_min_cycle_with_vertex_one(N, M, edges)