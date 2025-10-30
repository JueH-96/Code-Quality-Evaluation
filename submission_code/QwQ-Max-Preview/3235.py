class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = 10**18
        n = len(original)
        # Initialize distance matrix with infinity and 0 on the diagonal
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0
        
        # Populate the adjacency matrix with the given edges, keeping the minimum cost
        for o_char, c_char, co in zip(original, changed, cost):
            u = ord(o_char) - ord('a')
            v = ord(c_char) - ord('a')
            if dist[u][v] > co:
                dist[u][v] = co
        
        # Floyd-Warshall algorithm to compute all-pairs shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dist[u][v] == INF:
                return -1
            total_cost += dist[u][v]
        
        return total_cost