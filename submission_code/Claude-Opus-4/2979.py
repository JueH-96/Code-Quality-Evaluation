class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers by end point
        offers.sort(key=lambda x: x[1])
        
        m = len(offers)
        if m == 0:
            return 0
        
        # dp[i] = maximum profit considering offers 0 to i
        dp = [0] * m
        
        # Base case
        dp[0] = offers[0][2]
        
        for i in range(1, m):
            # Option 1: Don't take current offer
            profit_without_current = dp[i-1]
            
            # Option 2: Take current offer
            # Find the latest offer that doesn't overlap with current offer
            # Binary search for the latest offer whose end < current offer's start
            profit_with_current = offers[i][2]
            
            # Binary search
            left, right = 0, i - 1
            latest_non_overlapping = -1
            
            while left <= right:
                mid = (left + right) // 2
                if offers[mid][1] < offers[i][0]:
                    latest_non_overlapping = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            if latest_non_overlapping != -1:
                profit_with_current += dp[latest_non_overlapping]
            
            dp[i] = max(profit_without_current, profit_with_current)
        
        return dp[m-1]