class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        n = 26  # Number of lowercase letters
        
        # Initialize distance matrix
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0  # Cost to self is zero
        
        # Populate the adjacency matrix with the given edges (keep the minimum cost)
        for o, c, co in zip(original, changed, cost):
            x = ord(o) - ord('a')
            y = ord(c) - ord('a')
            if dist[x][y] > co:
                dist[x][y] = co
        
        # Floyd-Warshall algorithm to compute shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Calculate total minimum cost
        total_cost = 0
        for s, t in zip(source, target):
            if s == t:
                continue
            x = ord(s) - ord('a')
            y = ord(t) - ord('a')
            if dist[x][y] == INF:
                return -1  # No path exists
            total_cost += dist[x][y]
        
        return total_cost