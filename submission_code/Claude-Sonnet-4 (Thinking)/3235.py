class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize distance matrix
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        
        # Distance from a character to itself is 0
        for i in range(26):
            dist[i][i] = 0
        
        # Add edges from original to changed with given costs
        for i in range(len(original)):
            from_char = ord(original[i]) - ord('a')
            to_char = ord(changed[i]) - ord('a')
            dist[from_char][to_char] = min(dist[from_char][to_char], cost[i])
        
        # Floyd-Warshall algorithm
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate minimum cost
        total_cost = 0
        for i in range(len(source)):
            if source[i] != target[i]:
                from_char = ord(source[i]) - ord('a')
                to_char = ord(target[i]) - ord('a')
                if dist[from_char][to_char] == INF:
                    return -1
                total_cost += dist[from_char][to_char]
        
        return total_cost