import sys
from collections import defaultdict, deque

input = sys.stdin.read
def main():
    data = input().split()
    N = int(data[0])
    edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]
    
    if N == 3:
        print(0)
        return
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Function to find the center of the tree (or two centers if even number of nodes)
    def find_center(n, adj):
        # Find the farthest node from any arbitrary node using BFS
        def bfs(start):
            queue = deque([start])
            visited = [-1] * (n + 1)
            visited[start] = 0
            farthest_node = start
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if visited[neighbor] == -1:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
                        farthest_node = neighbor
            return farthest_node, visited
        
        # Find one farthest node from node 1 (arbitrary choice)
        farthest, _ = bfs(1)
        # Find the actual farthest node from the previously found farthest node
        farthest_from_farthest, distances = bfs(farthest)
        
        # Find the path from farthest to farthest_from_farthest
        max_distance = distances[farthest_from_farthest]
        path = []
        current = farthest_from_farthest
        while distances[current] != 0:
            path.append(current)
            for neighbor in adj[current]:
                if distances[neighbor] == distances[current] - 1:
                    current = neighbor
                    break
        path.append(current)
        
        # Center is the middle of the path
        if len(path) % 2 == 1:
            return [path[len(path) // 2]]
        else:
            return [path[len(path) // 2 - 1], path[len(path) // 2]]
    
    centers = find_center(N, adj)
    
    # Function to calculate the minimum deletions required to form a Snowflake Tree
    def calculate_min_deletions(center):
        # BFS to calculate subtree sizes and check if it can form a snowflake
        def bfs(center):
            queue = deque([center])
            visited = [-1] * (N + 1)
            visited[center] = 0
            parent = [-1] * (N + 1)
            subtree_sizes = [1] * (N + 1)
            
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if visited[neighbor] == -1:
                        visited[neighbor] = visited[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            # Calculate subtree sizes
            for node in reversed(queue):
                for neighbor in adj[node]:
                    if neighbor != parent[node]:
                        subtree_sizes[node] += subtree_sizes[neighbor]
            
            # Check if it can be a snowflake tree
            x = 0
            y = -1
            deletions = 0
            for neighbor in adj[center]:
                if parent[neighbor] == center:
                    branch_size = subtree_sizes[neighbor]
                    leaf_count = 0
                    for sub_neighbor in adj[neighbor]:
                        if parent[sub_neighbor] == neighbor:
                            leaf_count += 1
                            if subtree_sizes[sub_neighbor] != 1:
                                return float('inf')  # Not a valid snowflake structure
                    if y == -1:
                        y = leaf_count
                    elif y != leaf_count:
                        return float('inf')  # Inconsistent leaf count
                    x += 1
            
            # Calculate deletions
            deletions = N - (1 + x + x * y)
            return deletions
        
        return bfs(center)
    
    # Calculate the minimum deletions for each center
    min_deletions = float('inf')
    for center in centers:
        min_deletions = min(min_deletions, calculate_min_deletions(center))
    
    print(min_deletions)

main()