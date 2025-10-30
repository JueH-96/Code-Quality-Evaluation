import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    for i in range(1, 2 * N - 2, 2):
        edges.append((int(data[i]), int(data[i + 1])))
    C = list(map(int, data[2 * N - 1:]))
    
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Function to perform BFS and calculate distances from a given node
    def bfs(start):
        dist = [0] * (N + 1)
        visited = [False] * (N + 1)
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        
        return dist
    
    # Calculate the sum of distances for each node
    def calculate_f(node):
        dist = bfs(node)
        return sum(C[i - 1] * dist[i] for i in range(1, N + 1))
    
    # Find the minimum f(v)
    min_f = float('inf')
    for v in range(1, N + 1):
        min_f = min(min_f, calculate_f(v))
    
    print(min_f)

if __name__ == "__main__":
    main()