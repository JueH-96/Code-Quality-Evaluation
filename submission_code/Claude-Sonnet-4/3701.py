class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        def get_best_char_and_cost(start, end):
            """Find best character and cost for substring caption[start:end]"""
            group = caption[start:end]
            min_cost = float('inf')
            best_char = 'a'
            
            for target in 'abcdefghijklmnopqrstuvwxyz':
                cost = sum(abs(ord(c) - ord(target)) for c in group)
                if cost < min_cost or (cost == min_cost and target < best_char):
                    min_cost = cost
                    best_char = target
            
            return best_char, min_cost
        
        # dp[i] = (min_cost, result_string) for caption[0:i]
        dp = [None] * (n + 1)
        dp[0] = (0, "")
        
        for i in range(3, n + 1):
            min_cost = float('inf')
            best_result = None
            
            # Try all possible last groups of size >= 3
            for j in range(max(0, i - n), i - 2):  # j is start of last group
                if dp[j] is None:
                    continue
                
                prev_cost, prev_result = dp[j]
                char, group_cost = get_best_char_and_cost(j, i)
                
                total_cost = prev_cost + group_cost
                new_result = prev_result + char * (i - j)
                
                if (total_cost < min_cost or 
                    (total_cost == min_cost and (best_result is None or new_result < best_result))):
                    min_cost = total_cost
                    best_result = new_result
            
            if best_result is not None:
                dp[i] = (min_cost, best_result)
        
        return dp[n][1] if dp[n] is not None else ""