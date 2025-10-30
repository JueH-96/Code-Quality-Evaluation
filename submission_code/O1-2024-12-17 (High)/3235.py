class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        import math
        
        n = len(source)
        
        # There are 26 lowercase English letters, hence we create a 26x26 matrix for costs
        # Initialize all pairs with a large number (infinity) except dist[i][i] = 0
        INF = float('inf')
        dist = [[INF] * 26 for _ in range(26)]
        for i in range(26):
            dist[i][i] = 0

        # Fill direct transformation costs from original to changed
        # If multiple transformations for the same pair exist, choose the minimum cost
        for i in range(len(original)):
            from_char = ord(original[i]) - ord('a')
            to_char = ord(changed[i]) - ord('a')
            dist[from_char][to_char] = min(dist[from_char][to_char], cost[i])

        # Floyd-Warshall to find minimum cost from any character i to any character j
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # Now compute the total cost for transforming source to target character by character
        total_cost = 0
        for i in range(n):
            s_char = ord(source[i]) - ord('a')
            t_char = ord(target[i]) - ord('a')
            if dist[s_char][t_char] == INF:
                # Impossible to convert this character
                return -1
            total_cost += dist[s_char][t_char]

        return total_cost