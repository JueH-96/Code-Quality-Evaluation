class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # If source and target differ in length, it's impossible. (Though the problem states they have the same length.)
        if len(source) != len(target):
            return -1
        
        n = len(source)
        
        # There are only 26 lower-case letters. We will build a 26x26 cost matrix.
        # cost_matrix[i][j] represents the minimum cost to change character i -> j.
        INF = float('inf')
        cost_matrix = [[INF] * 26 for _ in range(26)]
        
        # Set cost of transforming a letter to itself = 0
        for i in range(26):
            cost_matrix[i][i] = 0
        
        # Populate the direct transformations from original to changed using the given cost array.
        # If there are multiple transformations for the same (original -> changed), keep the minimum cost.
        for o, c, z in zip(original, changed, cost):
            i_o = ord(o) - ord('a')
            i_c = ord(c) - ord('a')
            cost_matrix[i_o][i_c] = min(cost_matrix[i_o][i_c], z)
        
        # Use Floyd-Warshall to compute all-pairs shortest paths over the 26-letter graph.
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if cost_matrix[i][k] + cost_matrix[k][j] < cost_matrix[i][j]:
                        cost_matrix[i][j] = cost_matrix[i][k] + cost_matrix[k][j]
        
        # Now compute the total cost to transform source -> target.
        total_cost = 0
        for i in range(n):
            s_char = ord(source[i]) - ord('a')
            t_char = ord(target[i]) - ord('a')
            # If it's impossible to transform s_char to t_char, cost_matrix will remain INF.
            if cost_matrix[s_char][t_char] == INF:
                return -1
            total_cost += cost_matrix[s_char][t_char]
        
        return total_cost