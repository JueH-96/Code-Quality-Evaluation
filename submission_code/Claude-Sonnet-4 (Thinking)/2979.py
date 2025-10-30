class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        if not offers:
            return 0
            
        # Sort offers by end time
        offers.sort(key=lambda x: x[1])
        
        m = len(offers)
        dp = [0] * m
        
        # Base case
        dp[0] = offers[0][2]
        
        for i in range(1, m):
            # Option 1: Don't take current offer
            not_take = dp[i-1]
            
            # Option 2: Take current offer
            take = offers[i][2]
            
            # Find the latest non-conflicting offer
            j = i - 1
            while j >= 0 and offers[j][1] >= offers[i][0]:
                j -= 1
            
            if j >= 0:
                take += dp[j]
            
            dp[i] = max(not_take, take)
        
        return dp[m-1]