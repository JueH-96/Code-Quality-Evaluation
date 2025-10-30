from collections import deque

def solve():
    N = int(input())
    
    # Build adjacency list
    adj_list = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Calculate degree of each vertex
    degrees = [len(adj_list[i]) for i in range(N + 1)]
    
    # Find nodes with degree 2
    degree_2_nodes = [i for i in range(1, N + 1) if degrees[i] == 2]
    
    count = 0
    
    for i in range(len(degree_2_nodes)):
        for j in range(i + 1, len(degree_2_nodes)):
            u, v = degree_2_nodes[i], degree_2_nodes[j]
            
            # Skip if they're directly connected
            if v in adj_list[u]:
                continue
            
            # Find the path from u to v in the tree
            queue = deque([u])
            parent = [-1] * (N + 1)
            visited = [False] * (N + 1)
            visited[u] = True
            
            while queue:
                node = queue.popleft()
                if node == v:
                    break
                
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            # Reconstruct the path from v to u
            path = []
            current = v
            while current != -1:
                path.append(current)
                current = parent[current]
            
            # Check if all nodes on the path (excluding u and v) have degree 3
            path_valid = True
            for node in path[1:-1]:  # Exclude v and u
                if degrees[node] != 3:
                    path_valid = False
                    break
            
            if path_valid:
                count += 1
    
    return count

print(solve())