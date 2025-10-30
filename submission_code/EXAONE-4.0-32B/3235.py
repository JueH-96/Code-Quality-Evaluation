class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        INF = 10**18
        dp = [[INF] * 26 for _ in range(26)]
        
        for i in range(26):
            dp[i][i] = 0
            
        for o_char, c_char, co in zip(original, changed, cost):
            u = ord(o_char) - ord('a')
            v = ord(c_char) - ord('a')
            if co < dp[u][v]:
                dp[u][v] = co
                
        for k in range(26):
            for i in range(26):
                if dp[i][k] == INF:
                    continue
                for j in range(26):
                    new_cost = dp[i][k] + dp[k][j]
                    if new_cost < dp[i][j]:
                        dp[i][j] = new_cost
                        
        total_cost = 0
        n = len(source)
        for i in range(n):
            s_char = source[i]
            t_char = target[i]
            if s_char == t_char:
                continue
            u = ord(s_char) - ord('a')
            v = ord(t_char) - ord('a')
            if dp[u][v] == INF:
                return -1
            total_cost += dp[u][v]
            
        return total_cost