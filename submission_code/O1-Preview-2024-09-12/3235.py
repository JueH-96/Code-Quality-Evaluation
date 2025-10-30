class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Initialize the distance matrix with infinity
        import sys
        dist = [[float('inf')] * 26 for _ in range(26)]
        
        # Set distance from a node to itself as 0
        for i in range(26):
            dist[i][i] = 0
        
        # Build the adjacency matrix with given transformations
        for o, c, w in zip(original, changed, cost):
            u = ord(o) - ord('a')
            v = ord(c) - ord('a')
            if dist[u][v] > w:
                dist[u][v] = w
         
        # Floyd-Warshall algorithm to find shortest paths between all pairs of nodes
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
            if dist[u][v] == float('inf'):
                return -1
            total_cost += dist[u][v]
        return int(total_cost)