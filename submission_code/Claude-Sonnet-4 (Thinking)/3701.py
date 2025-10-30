class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        # dp[i] = (min_cost, best_result_string)
        dp = [None] * (n + 1)
        dp[0] = (0, "")
        
        for i in range(3, n + 1):
            for j in range(i - 2):  # segment length at least 3
                if dp[j] is None:
                    continue
                
                segment = caption[j:i]
                sorted_segment = sorted(segment)
                segment_length = len(segment)
                
                # Find the optimal character (median)
                if segment_length % 2 == 1:
                    best_char = sorted_segment[segment_length // 2]
                else:
                    best_char = sorted_segment[segment_length // 2 - 1]  # choose the smaller middle character
                
                best_cost = sum(abs(ord(ch) - ord(best_char)) for ch in segment)
                
                total_cost = dp[j][0] + best_cost
                result_string = dp[j][1] + best_char * segment_length
                
                if dp[i] is None or total_cost < dp[i][0] or (total_cost == dp[i][0] and result_string < dp[i][1]):
                    dp[i] = (total_cost, result_string)
        
        return dp[n][1] if dp[n] is not None else ""