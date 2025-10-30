class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers by end position
        offers.sort(key=lambda x: x[1])
        
        m = len(offers)
        if m == 0:
            return 0
        
        # dp[i] represents max gold using offers 0 to i
        dp = [0] * m
        dp[0] = offers[0][2]  # First offer's gold
        
        def findLatestNonConflicting(i):
            # Find the latest offer j such that offers[j][1] < offers[i][0]
            # Using binary search
            left, right = 0, i - 1
            result = -1
            
            while left <= right:
                mid = (left + right) // 2
                if offers[mid][1] < offers[i][0]:
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        
        for i in range(1, m):
            # Option 1: Don't take current offer
            dp[i] = dp[i-1]
            
            # Option 2: Take current offer
            latest_non_conflict = findLatestNonConflicting(i)
            current_gold = offers[i][2]
            
            if latest_non_conflict == -1:
                # No previous non-conflicting offer
                dp[i] = max(dp[i], current_gold)
            else:
                dp[i] = max(dp[i], dp[latest_non_conflict] + current_gold)
        
        return dp[m-1]