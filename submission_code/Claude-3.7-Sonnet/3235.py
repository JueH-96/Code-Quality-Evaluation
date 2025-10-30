class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize distance matrix with infinity
        dist = [[float('inf')] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0  # Cost to transform a character to itself is 0
        
        # Add edges to the graph based on provided operations
        for i in range(len(original)):
            src_idx = ord(original[i]) - ord('a')
            dest_idx = ord(changed[i]) - ord('a')
            # If there are multiple operations for the same transformation, take the minimum cost
            dist[src_idx][dest_idx] = min(dist[src_idx][dest_idx], cost[i])
        
        # Floyd-Warshall algorithm to find shortest paths between all pairs
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Calculate total cost to transform source to target
        total_cost = 0
        for i in range(len(source)):
            if source[i] != target[i]:
                src_idx = ord(source[i]) - ord('a')
                dest_idx = ord(target[i]) - ord('a')
                
                # If no path exists to transform this character, return -1
                if dist[src_idx][dest_idx] == float('inf'):
                    return -1
                
                total_cost += dist[src_idx][dest_idx]
        
        return total_cost