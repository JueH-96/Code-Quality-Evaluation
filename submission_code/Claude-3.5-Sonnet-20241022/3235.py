class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create adjacency matrix for min costs between chars
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        
        # Initialize diagonal with 0 cost
        for i in range(26):
            dist[i][i] = 0
            
        # Build initial graph from given transformations
        for i in range(len(original)):
            orig = ord(original[i]) - ord('a')
            chg = ord(changed[i]) - ord('a')
            # Take minimum cost if multiple paths exist
            dist[orig][chg] = min(dist[orig][chg], cost[i])
        
        # Floyd-Warshall to find shortest paths between all pairs
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        # Calculate minimum cost to transform source to target
        total_cost = 0
        for i in range(len(source)):
            if source[i] == target[i]:
                continue
                
            src_idx = ord(source[i]) - ord('a')
            tgt_idx = ord(target[i]) - ord('a')
            
            if dist[src_idx][tgt_idx] == INF:
                return -1
                
            total_cost += dist[src_idx][tgt_idx]
            
        return total_cost