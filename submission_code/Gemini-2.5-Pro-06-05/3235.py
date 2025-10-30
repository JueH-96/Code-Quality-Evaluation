class Solution:
    def minimumCost(self, source: str, target: str, original: list[str], changed: list[str], cost: list[int]) -> int:
        
        N_CHARS = 26
        inf = float('inf')
        
        # dist[i][j] will store the minimum cost to change character i to character j.
        # This is an adjacency matrix for our graph of characters.
        dist = [[inf] * N_CHARS for _ in range(N_CHARS)]
        
        # The cost to change a character to itself is 0.
        for i in range(N_CHARS):
            dist[i][i] = 0
            
        # Populate the matrix with initial direct transformation costs.
        # ord(char) - ord('a') maps characters 'a'-'z' to indices 0-25.
        for i in range(len(original)):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            c = cost[i]
            # If multiple transformations exist for the same pair, take the minimum cost.
            dist[u][v] = min(dist[u][v], c)
            
        # Use Floyd-Warshall algorithm to find all-pairs shortest paths.
        # This accounts for multi-step transformations (e.g., a -> c -> b).
        for k in range(N_CHARS):
            for i in range(N_CHARS):
                for j in range(N_CHARS):
                    # If a path through character k is shorter, update the distance.
                    if dist[i][k] != inf and dist[k][j] != inf:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                        
        # Calculate the total cost to convert the source string to the target string.
        total_cost = 0
        for i in range(len(source)):
            # If characters are the same, no cost is incurred for this position.
            if source[i] == target[i]:
                continue
            
            s_idx = ord(source[i]) - ord('a')
            t_idx = ord(target[i]) - ord('a')
            
            transformation_cost = dist[s_idx][t_idx]
            
            # If the cost is infinity, the transformation is impossible.
            if transformation_cost == inf:
                return -1
                
            total_cost += transformation_cost
            
        return int(total_cost)