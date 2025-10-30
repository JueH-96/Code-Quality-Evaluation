class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # dp1[i] = length of the longest non-decreasing subarray ending at i, 
        #          if we choose nums1[i] for nums3[i]
        # dp2[i] = length of the longest non-decreasing subarray ending at i,
        #          if we choose nums2[i] for nums3[i]
        dp1 = [0] * n
        dp2 = [0] * n
        
        dp1[0] = 1
        dp2[0] = 1
        answer = 1
        
        for i in range(1, n):
            # Default length is 1 if we cannot extend from i-1
            dp1[i] = 1
            dp2[i] = 1
            
            # If previous choice was nums1[i-1], can we extend to nums1[i]?
            if nums1[i - 1] <= nums1[i]:
                dp1[i] = max(dp1[i], dp1[i - 1] + 1)
            # If previous choice was nums2[i-1], can we extend to nums1[i]?
            if nums2[i - 1] <= nums1[i]:
                dp1[i] = max(dp1[i], dp2[i - 1] + 1)
            
            # If previous choice was nums1[i-1], can we extend to nums2[i]?
            if nums1[i - 1] <= nums2[i]:
                dp2[i] = max(dp2[i], dp1[i - 1] + 1)
            # If previous choice was nums2[i-1], can we extend to nums2[i]?
            if nums2[i - 1] <= nums2[i]:
                dp2[i] = max(dp2[i], dp2[i - 1] + 1)
            
            # Update the global answer
            answer = max(answer, dp1[i], dp2[i])
        
        return answer