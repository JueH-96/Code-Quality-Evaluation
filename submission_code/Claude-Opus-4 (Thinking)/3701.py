class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        # dp[i] = (min_cost, resulting_string) for making the first i characters good
        dp = [(float('inf'), "")] * (n + 1)
        dp[0] = (0, "")
        
        for i in range(3, n + 1):
            for j in range(i - 2):  # j goes from 0 to i-3
                # Consider group from position j to i-1 (inclusive)
                # Find the best character to convert this group to
                best_cost = float('inf')
                best_char = ''
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    cost = sum(abs(ord(caption[k]) - ord(c)) for k in range(j, i))
                    if cost < best_cost or (cost == best_cost and c < best_char):
                        best_cost = cost
                        best_char = c
                
                total_cost = dp[j][0] + best_cost
                new_string = dp[j][1] + best_char * (i - j)
                
                if total_cost < dp[i][0] or (total_cost == dp[i][0] and new_string < dp[i][1]):
                    dp[i] = (total_cost, new_string)
        
        return dp[n][1] if dp[n][0] != float('inf') else ""