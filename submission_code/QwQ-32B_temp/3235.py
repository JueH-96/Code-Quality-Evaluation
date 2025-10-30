class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n_chars = 26
        INF = float('inf')
        
        # Initialize distance matrix
        dist = [[INF] * n_chars for _ in range(n_chars)]
        for i in range(n_chars):
            dist[i][i] = 0
        
        # Update with given edges, keeping the minimum cost for each pair
        for i in range(len(original)):
            u = ord(original[i]) - ord('a')
            v = ord(changed[i]) - ord('a')
            c = cost[i]
            if c < dist[u][v]:
                dist[u][v] = c
        
        # Floyd-Warshall algorithm to find all pairs shortest paths
        for k in range(n_chars):
            for i in range(n_chars):
                for j in range(n_chars):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        for s, t in zip(source, target):
            if s == t:
                continue
            u = ord(s) - ord('a')
            v = ord(t) - ord('a')
            if dist[u][v] == INF:
                return -1
            total_cost += dist[u][v]
        
        return total_cost