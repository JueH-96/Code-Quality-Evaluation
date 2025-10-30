class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on the end point, then start point
        offers.sort(key=lambda x: (x[1], x[0]))
        
        # dp[i] will store the maximum gold we can get by considering houses from 0 to i
        dp = [0] * n
        
        # Initialize a pointer for the last non-overlapping offer
        last_non_overlap = [-1] * n
        
        # Precompute the last non-overlapping index for each offer
        j = -1
        for i in range(n):
            while j + 1 < len(offers) and offers[j + 1][1] < i:
                j += 1
            last_non_overlap[i] = j
        
        for i in range(n):
            # If no house is sold till i, then dp[i] is just dp[i-1] (or 0 if i == 0)
            dp[i] = dp[i-1] if i > 0 else 0
            
            # Process each offer ending at i
            for j in range(last_non_overlap[i-1] + 1, last_non_overlap[i] + 1):
                start, end, gold = offers[j]
                if start == 0:
                    # If the offer starts from the first house, no previous houses to consider
                    dp[i] = max(dp[i], gold)
                else:
                    # Otherwise, add the best profit before this offer starts
                    dp[i] = max(dp[i], dp[start-1] + gold)
        
        # The answer for all houses from 0 to n-1 is stored in dp[n-1]
        return dp[n-1]