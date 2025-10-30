from collections import deque

def solve():
    N, Q = map(int, input().split())
    
    # Initialize graph as adjacency list
    graph = [set() for _ in range(N+1)]
    
    # Process queries
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:  # Add edge
            u, v = query[1], query[2]
            graph[u].add(v)
            graph[v].add(u)  # Undirected graph
        
        else:  # Find k-th largest vertex
            v, k = query[1], query[2]
            
            # Find all vertices in the same connected component as v using BFS
            visited = [False] * (N+1)
            connected_vertices = []
            
            queue = deque([v])
            visited[v] = True
            connected_vertices.append(v)
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        connected_vertices.append(neighbor)
                        queue.append(neighbor)
            
            # Sort in descending order
            connected_vertices.sort(reverse=True)
            
            # Output k-th largest if it exists
            if k <= len(connected_vertices):
                print(connected_vertices[k-1])
            else:
                print(-1)

if __name__ == "__main__":
    solve()