class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Base case - single element array is always non-decreasing
        if n == 1:
            return 1
        
        # dp1[i] = length of longest non-decreasing subarray ending at i, choosing nums1[i]
        # dp2[i] = length of longest non-decreasing subarray ending at i, choosing nums2[i]
        # We can optimize to use only variables since we only need previous values
        dp1_prev = 1
        dp2_prev = 1
        
        max_length = 1
        
        for i in range(1, n):
            # Current dp values (default to 1 if we can't extend previous subarrays)
            dp1_curr = 1
            dp2_curr = 1
            
            # If we choose nums1[i], check if we can extend from previous positions
            if nums1[i] >= nums1[i-1]:
                dp1_curr = max(dp1_curr, dp1_prev + 1)
            if nums1[i] >= nums2[i-1]:
                dp1_curr = max(dp1_curr, dp2_prev + 1)
            
            # If we choose nums2[i], check if we can extend from previous positions
            if nums2[i] >= nums1[i-1]:
                dp2_curr = max(dp2_curr, dp1_prev + 1)
            if nums2[i] >= nums2[i-1]:
                dp2_curr = max(dp2_curr, dp2_prev + 1)
            
            # Update maximum length
            max_length = max(max_length, dp1_curr, dp2_curr)
            
            # Update previous values for next iteration
            dp1_prev = dp1_curr
            dp2_prev = dp2_curr
        
        return max_length