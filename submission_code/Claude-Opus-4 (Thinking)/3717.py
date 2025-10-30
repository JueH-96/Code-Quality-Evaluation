class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        
        # Precompute cost for each subarray of size x
        cost = []
        for i in range(n - x + 1):
            subarray = sorted(nums[i:i+x])
            # Median minimizes sum of absolute deviations
            median = subarray[x // 2] if x % 2 == 1 else subarray[x // 2 - 1]
            # Calculate cost to make all elements equal to median
            c = sum(abs(nums[i+j] - median) for j in range(x))
            cost.append(c)
        
        # DP: dp[i][j] = min cost to select j non-overlapping subarrays from nums[0:i]
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        
        # Base case: selecting 0 subarrays costs 0
        for i in range(n + 1):
            dp[i][0] = 0
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # Option 1: Don't include a subarray ending at position i-1
                dp[i][j] = dp[i-1][j]
                
                # Option 2: Include a subarray ending at position i-1
                if i >= x:
                    dp[i][j] = min(dp[i][j], dp[i-x][j-1] + cost[i-x])
        
        return dp[n][k]