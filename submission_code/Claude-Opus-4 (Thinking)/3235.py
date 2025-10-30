class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize distance matrix for 26 letters
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        
        # Distance from each letter to itself is 0
        for i in range(26):
            dist[i][i] = 0
        
        # Add edges from the given transformations
        for i in range(len(original)):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            # Take minimum cost if there are multiple edges
            dist[u][v] = min(dist[u][v], cost[i])
        
        # Floyd-Warshall algorithm
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate total cost
        total_cost = 0
        for i in range(len(source)):
            if source[i] == target[i]:
                continue
            
            s = ord(source[i]) - ord('a')
            t = ord(target[i]) - ord('a')
            
            if dist[s][t] == INF:
                return -1
            
            total_cost += dist[s][t]
        
        return total_cost