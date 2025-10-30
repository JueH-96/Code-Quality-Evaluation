from typing import List

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = float('inf')
        # Create a 26x26 DP matrix for minimum conversion cost between any two letters.
        dp = [[INF] * 26 for _ in range(26)]
        
        # Zero cost to remain the same letter.
        for i in range(26):
            dp[i][i] = 0
        
        # Set direct conversion costs, taking the minimum if there are duplicates.
        for o, ch, c in zip(original, changed, cost):
            u = ord(o) - ord('a')
            v = ord(ch) - ord('a')
            dp[u][v] = min(dp[u][v], c)
        
        # Apply Floyd-Warshall to compute all-pairs shortest paths.
        for k in range(26):
            for i in range(26):
                if dp[i][k] == INF:
                    continue
                for j in range(26):
                    if dp[k][j] != INF:
                        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
        
        total_cost = 0
        # For each position, if source and target differ, add the minimum conversion cost.
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dp[u][v] == INF:
                # If conversion from s_char to t_char is impossible.
                return -1
            total_cost += dp[u][v]
        
        return total_cost