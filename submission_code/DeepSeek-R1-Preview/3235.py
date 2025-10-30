class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        # Initialize the distance matrix with infinity
        distance = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            distance[i][i] = 0
        
        # Populate the direct edges with the minimum cost
        for o_char, c_char, co in zip(original, changed, cost):
            o = ord(o_char) - ord('a')
            c = ord(c_char) - ord('a')
            if distance[o][c] > co:
                distance[o][c] = co
        
        # Floyd-Warshall algorithm to find the shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if distance[i][k] + distance[k][j] < distance[i][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
        
        total_cost = 0
        # Calculate the total minimum cost
        for s, t in zip(source, target):
            if s == t:
                continue
            s_idx = ord(s) - ord('a')
            t_idx = ord(t) - ord('a')
            if distance[s_idx][t_idx] == INF:
                return -1
            total_cost += distance[s_idx][t_idx]
        
        return total_cost