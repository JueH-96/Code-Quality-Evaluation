class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 1:
            return 1
            
        # dp[i][j] represents length ending at i using nums[j]
        # j=0 means using nums1[i], j=1 means using nums2[i]
        dp = [[1]*2 for _ in range(n)]
        max_len = 1
        
        for i in range(1, n):
            # For nums1[i]
            if nums1[i] >= nums1[i-1]:
                dp[i][0] = dp[i-1][0] + 1
            if nums1[i] >= nums2[i-1]:
                dp[i][0] = max(dp[i][0], dp[i-1][1] + 1)
                
            # For nums2[i]    
            if nums2[i] >= nums1[i-1]:
                dp[i][1] = dp[i-1][0] + 1
            if nums2[i] >= nums2[i-1]:
                dp[i][1] = max(dp[i][1], dp[i-1][1] + 1)
                
            max_len = max(max_len, dp[i][0], dp[i][1])
            
        return max_len