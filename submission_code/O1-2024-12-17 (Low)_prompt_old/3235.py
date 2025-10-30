class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        import math
        
        # Edge case: if source and target have different lengths, no valid solution.
        # (Though problem states they have the same length.)
        n = len(source)
        if n != len(target):
            return -1
        
        # Map characters 'a'..'z' to indices 0..25
        def char_to_index(c: str) -> int:
            return ord(c) - ord('a')
        
        # Initialize distance array for Floyd-Warshall: 26 x 26
        # dist[i][j] = cost to transform from i to j.
        # Start with infinity except for dist[i][i] = 0
        INF = math.inf
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0
        
        # Populate direct transformation costs from given arrays
        # We can have multiple edges from a char x to y, so we take the minimum cost
        for i in range(len(original)):
            x = char_to_index(original[i])
            y = char_to_index(changed[i])
            c = cost[i]
            dist[x][y] = min(dist[x][y], c)
        
        # Floyd-Warshall to find minimum cost for all pairs of chars
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Now compute total cost for transforming source to target
        total_cost = 0
        for i in range(n):
            s_char = char_to_index(source[i])
            t_char = char_to_index(target[i])
            if dist[s_char][t_char] == INF:
                return -1  # impossible
            total_cost += dist[s_char][t_char]
        
        return total_cost