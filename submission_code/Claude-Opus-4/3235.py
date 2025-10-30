class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize distance matrix with infinity
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        
        # Distance from a character to itself is 0
        for i in range(26):
            dist[i][i] = 0
        
        # Build the graph with given transformations
        for i in range(len(original)):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            # Take minimum cost if multiple transformations exist
            dist[u][v] = min(dist[u][v], cost[i])
        
        # Floyd-Warshall algorithm to find shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate total cost
        total_cost = 0
        for i in range(len(source)):
            if source[i] == target[i]:
                continue
            
            src_char = ord(source[i]) - ord('a')
            tgt_char = ord(target[i]) - ord('a')
            
            if dist[src_char][tgt_char] == INF:
                return -1
            
            total_cost += dist[src_char][tgt_char]
        
        return total_cost