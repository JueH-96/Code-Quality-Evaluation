class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        import math
        
        # If source and target are the same, cost is 0
        if source == target:
            return 0
        
        n = len(source)
        
        # We only care about cost to transform from one character 'a'..'z' to another.
        # Build a 26x26 cost matrix; matrix[u][v] = min cost to transform
        # from character u to character v, where u, v are 0-based indices for 'a'..'z'.
        INF = 10**15
        dist = [[INF]*26 for _ in range(26)]
        
        # Cost to stay the same is zero (no change needed).
        for i in range(26):
            dist[i][i] = 0
        
        # Fill cost transitions from the input arrays
        # original[i] -> changed[i] at cost[i].
        for x, y, cst in zip(original, changed, cost):
            u = ord(x) - ord('a')
            v = ord(y) - ord('a')
            dist[u][v] = min(dist[u][v], cst)  # take the minimum cost if multiple edges exist
        
        # Floyd-Warshall to find minimum cost for any single-char transform
        for k in range(26):
            for i in range(26):
                for j in range(26):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        total_cost = 0
        
        for i in range(n):
            s_char = ord(source[i]) - ord('a')
            t_char = ord(target[i]) - ord('a')
            # If direct or indirect transformation is not possible, dist[s_char][t_char] would be INF.
            if dist[s_char][t_char] == INF:
                return -1
            total_cost += dist[s_char][t_char]
        
        return total_cost