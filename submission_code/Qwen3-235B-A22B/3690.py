class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        low = 1
        high = n
        ans = n
        
        def cost_for_L(L):
            INF = float('inf')
            # dp[b][k] represents min flips to end at current position with run of type b and length k
            dp = [[INF] * (L + 2) for _ in range(2)]  # 0-based b (0 or 1), k up to L
            
            # Initialize first character
            first_char = int(s[0])
            dp[0][1] = 0 if first_char == 0 else 1
            dp[1][1] = 0 if first_char == 1 else 1
            
            for i in range(1, n):
                new_dp = [[INF] * (L + 2) for _ in range(2)]
                current_char = int(s[i])
                for b in range(2):
                    for k in range(1, L + 1):
                        prev_cost = dp[b][k]
                        if prev_cost == INF:
                            continue
                        
                        # Option 1: Extend the current run if possible
                        if k < L:
                            cost = 0 if (b == current_char) else 1
                            if new_dp[b][k + 1] > prev_cost + cost:
                                new_dp[b][k + 1] = prev_cost + cost
                        
                        # Option 2: Start a new run of 1 - b
                        new_b = 1 - b
                        flip_cost = 0 if (new_b == current_char) else 1
                        if new_dp[new_b][1] > prev_cost + flip_cost:
                            new_dp[new_b][1] = prev_cost + flip_cost
                
                dp = new_dp  # Move to next character
            
            min_total = INF
            for b in range(2):
                for k in range(1, L + 1):
                    if dp[b][k] < min_total:
                        min_total = dp[b][k]
            return min_total
        
        # Binary search to find minimal L
        while low <= high:
            mid = (low + high) // 2
            required = cost_for_L(mid)
            if required <= numOps:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans