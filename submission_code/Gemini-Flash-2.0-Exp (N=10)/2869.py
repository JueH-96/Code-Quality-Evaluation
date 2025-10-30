class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        dp = [[0, 0] for _ in range(n)]
        dp[0][0] = 1
        dp[0][1] = 1
        
        max_len = 1
        
        for i in range(1, n):
            
            len1 = 1
            len2 = 1
            
            if nums1[i] >= nums1[i-1]:
                len1 = max(len1, dp[i-1][0] + 1)
            if nums1[i] >= nums2[i-1]:
                len1 = max(len1, dp[i-1][1] + 1)
            
            if nums2[i] >= nums1[i-1]:
                len2 = max(len2, dp[i-1][0] + 1)
            if nums2[i] >= nums2[i-1]:
                len2 = max(len2, dp[i-1][1] + 1)
            
            dp[i][0] = len1
            dp[i][1] = len2
            
            max_len = max(max_len, len1, len2)
            
        return max_len