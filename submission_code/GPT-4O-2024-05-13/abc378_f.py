# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N-1)]
    
    # Create adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Calculate degrees
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(graph[u])
    
    # Find all pairs of vertices that are not directly connected
    not_connected = set()
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if v not in graph[u]:
                not_connected.add((u, v))
    
    def bfs_cycle_check(start, end):
        # BFS to find the cycle
        parent = [-1] * (N + 1)
        visited = [False] * (N + 1)
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a cycle
                    cycle = set()
                    x = node
                    while x != -1:
                        cycle.add(x)
                        x = parent[x]
                    x = neighbor
                    while x != -1:
                        cycle.add(x)
                        x = parent[x]
                    return cycle
        
        return set()
    
    valid_cycles = 0
    
    for u, v in not_connected:
        # Temporarily add the edge (u, v)
        graph[u].append(v)
        graph[v].append(u)
        
        # Check if it forms a valid cycle
        cycle = bfs_cycle_check(u, v)
        if cycle and all(degree[node] == 2 for node in cycle):
            valid_cycles += 1
        
        # Remove the edge (u, v)
        graph[u].remove(v)
        graph[v].remove(u)
    
    print(valid_cycles)

if __name__ == "__main__":
    main()