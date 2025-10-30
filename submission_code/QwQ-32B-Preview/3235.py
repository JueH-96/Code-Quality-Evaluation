class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        import sys
        INF = float('inf')
        
        # Initialize the cost matrix
        matrix = [[INF for _ in range(26)] for _ in range(26)]
        for i in range(26):
            matrix[i][i] = 0
        
        # Update the matrix with given transformations
        for o, c, co in zip(original, changed, cost):
            o_idx = ord(o) - ord('a')
            c_idx = ord(c) - ord('a')
            if matrix[o_idx][c_idx] > co:
                matrix[o_idx][c_idx] = co
        
        # Floyd-Warshall algorithm to compute all-pairs shortest paths
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if matrix[i][k] != INF and matrix[k][j] != INF:
                        if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                            matrix[i][j] = matrix[i][k] + matrix[k][j]
        
        total_cost = 0
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            s_idx = ord(s_char) - ord('a')
            t_idx = ord(t_char) - ord('a')
            min_cost = matrix[s_idx][t_idx]
            if min_cost == INF:
                return -1
            total_cost += min_cost
        
        return total_cost