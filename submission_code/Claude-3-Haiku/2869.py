class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp = [1] * n  # Initialize the dp array with 1s
        
        for i in range(1, n):
            # If we can choose nums1[i], update the dp value
            if nums1[i] >= nums1[i-1]:
                dp[i] = max(dp[i], dp[i-1] + 1)
            # If we can choose nums2[i], update the dp value
            if nums2[i] >= nums1[i-1]:
                dp[i] = max(dp[i], dp[i-1] + 1)
            
        return max(dp)