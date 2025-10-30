class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        n = 26  # Number of lowercase letters
        distance = [[INF] * n for _ in range(n)]
        
        # Initialize the distance matrix with 0 for same nodes
        for i in range(n):
            distance[i][i] = 0
        
        # Populate the initial distance matrix with the given transitions
        for j in range(len(original)):
            x = original[j]
            y = changed[j]
            c = cost[j]
            x_idx = ord(x) - ord('a')
            y_idx = ord(y) - ord('a')
            if distance[x_idx][y_idx] > c:
                distance[x_idx][y_idx] = c
        
        # Floyd-Warshall algorithm to find the shortest paths between all pairs
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distance[i][k] + distance[k][j] < distance[i][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            s_idx = ord(s_char) - ord('a')
            t_idx = ord(t_char) - ord('a')
            if distance[s_idx][t_idx] == INF:
                return -1
            total_cost += distance[s_idx][t_idx]
        
        return total_cost