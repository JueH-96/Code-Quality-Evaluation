class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Check if initial sum is already <= x
        if sum(nums1) <= x:
            return 0
        
        # Create pairs of (nums2[i], nums1[i]) and sort by nums2[i]
        # This helps us prioritize which elements to zero out
        pairs = sorted(zip(nums2, nums1))
        
        # dp[i][j] = maximum reduction we can achieve using first i elements with j operations
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            b, a = pairs[i - 1]  # nums2[i-1], nums1[i-1] after sorting
            for j in range(n + 1):
                # Option 1: Don't use this element
                dp[i][j] = dp[i - 1][j]
                
                # Option 2: Use this element at time j
                if j > 0:
                    # If we zero out at time j, we save:
                    # - The initial value: a
                    # - The accumulated value: b * j
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + a + b * j)
        
        # Check each time t
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        
        for t in range(n + 1):
            # Total sum at time t without any operations would be:
            # sum(nums1) + t * sum(nums2)
            # With operations, we can reduce by dp[n][t]
            total = sum1 + t * sum2 - dp[n][t]
            if total <= x:
                return t
        
        return -1