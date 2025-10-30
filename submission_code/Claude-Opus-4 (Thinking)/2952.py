class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # If initial sum is already <= x
        if sum(nums1) <= x:
            return 0
        
        # Pair each element with its index and sort by nums2 value
        pairs = sorted(zip(nums2, nums1))
        
        # dp[i][j] = maximum reduction achievable using first i elements with j operations
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            growth_rate, initial_val = pairs[i - 1]
            for j in range(i + 1):
                # Option 1: Don't reset element i
                dp[i][j] = dp[i - 1][j]
                
                # Option 2: Reset element i at time j
                if j > 0:
                    # Reduction = initial_val + j * growth_rate
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + initial_val + j * growth_rate)
        
        # Check each possible time t
        initial_sum = sum(nums1)
        total_growth_rate = sum(nums2)
        
        for t in range(n + 1):
            # Sum after t seconds with optimal resets
            current_sum = initial_sum + t * total_growth_rate - dp[n][t]
            if current_sum <= x:
                return t
        
        return -1