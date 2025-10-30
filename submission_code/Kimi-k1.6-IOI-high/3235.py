class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        n = 26
        # Initialize the cost matrix with infinity and 0 on the diagonal
        matrix = [[INF] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 0
        
        # Populate the matrix with given costs
        for o_char, c_char, c_cost in zip(original, changed, cost):
            u = ord(o_char) - ord('a')
            v = ord(c_char) - ord('a')
            if matrix[u][v] > c_cost:
                matrix[u][v] = c_cost
        
        # Floyd-Warshall algorithm to compute all-pairs shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if matrix[i][k] + matrix[k][j] < matrix[i][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if matrix[u][v] == INF:
                return -1
            total_cost += matrix[u][v]
        
        return total_cost