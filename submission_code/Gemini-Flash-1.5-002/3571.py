class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        # Create a graph where each node represents a coordinate.
        # An edge exists between two nodes if they form an increasing path.
        graph = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and coordinates[i][0] < coordinates[j][0] and coordinates[i][1] < coordinates[j][1]:
                    graph[i].append(j)

        # Perform Depth-First Search (DFS) starting from node k.
        max_length = 0
        visited = set()

        def dfs(node, current_length):
            nonlocal max_length
            max_length = max(max_length, current_length)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_length + 1)
            visited.remove(node)

        dfs(k, 1)
        return max_length