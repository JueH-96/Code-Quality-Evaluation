class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize distance matrix with infinity
        # We have 26 lowercase letters
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        
        # Distance from a character to itself is 0
        for i in range(26):
            dist[i][i] = 0
        
        # Build the graph with given transformations
        for i in range(len(original)):
            from_char = ord(original[i]) - ord('a')
            to_char = ord(changed[i]) - ord('a')
            # Take minimum cost if there are multiple edges between same characters
            dist[from_char][to_char] = min(dist[from_char][to_char], cost[i])
        
        # Floyd-Warshall algorithm to find shortest paths between all pairs
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate total cost
        total_cost = 0
        for i in range(len(source)):
            if source[i] != target[i]:
                from_char = ord(source[i]) - ord('a')
                to_char = ord(target[i]) - ord('a')
                
                if dist[from_char][to_char] == INF:
                    return -1  # Impossible to transform
                
                total_cost += dist[from_char][to_char]
        
        return total_cost