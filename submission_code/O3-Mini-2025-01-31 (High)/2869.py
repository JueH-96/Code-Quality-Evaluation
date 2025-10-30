from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # Base case: if there's only one element, the longest non-decreasing subarray is 1.
        dp0, dp1 = 1, 1  # dp0: longest subarray ending at i when choosing nums1[i]
                         # dp1: longest subarray ending at i when choosing nums2[i]
        ans = 1
        
        for i in range(1, n):
            # Start new subarray from index i if extension isn't possible; hence the initial 1.
            new0, new1 = 1, 1
            
            # Option 1: choose nums1[i] as the i-th element
            if nums1[i] >= nums1[i - 1]:
                new0 = max(new0, dp0 + 1)
            if nums1[i] >= nums2[i - 1]:
                new0 = max(new0, dp1 + 1)
            
            # Option 2: choose nums2[i] as the i-th element
            if nums2[i] >= nums1[i - 1]:
                new1 = max(new1, dp0 + 1)
            if nums2[i] >= nums2[i - 1]:
                new1 = max(new1, dp1 + 1)
            
            # Update dp values for the next iteration.
            dp0, dp1 = new0, new1
            ans = max(ans, dp0, dp1)
        
        return ans