class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Sort indices by nums2 values
        indices = list(range(n))
        indices.sort(key=lambda i: nums2[i])
        
        # dp[i][j] = minimum sum of first i elements after j seconds with j resets
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            idx = indices[i - 1]
            for j in range(i + 1):
                # Don't reset element idx
                dp[i][j] = dp[i - 1][j] + nums1[idx] + nums2[idx] * j
                
                # Reset element idx at second j (if j > 0)
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
        
        # Try each possible time from 0 to n
        for t in range(n + 1):
            if dp[n][t] <= x:
                return t
        
        return -1