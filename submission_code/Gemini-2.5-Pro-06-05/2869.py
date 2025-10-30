from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # dp1: length of the longest non-decreasing subarray ending at the previous index, using nums1.
        # dp2: length of the longest non-decreasing subarray ending at the previous index, using nums2.
        # Base case for index 0: A single element forms a subarray of length 1.
        dp1 = 1
        dp2 = 1
        max_length = 1
        
        for i in range(1, n):
            # Calculate the length of the non-decreasing subarray ending at index i
            # if we choose nums1[i].
            # A new subarray of length 1 can always be started.
            temp_dp1 = 1
            if nums1[i] >= nums1[i-1]:
                temp_dp1 = max(temp_dp1, dp1 + 1)
            if nums1[i] >= nums2[i-1]:
                temp_dp1 = max(temp_dp1, dp2 + 1)
            
            # Calculate the length of the non-decreasing subarray ending at index i
            # if we choose nums2[i].
            temp_dp2 = 1
            if nums2[i] >= nums1[i-1]:
                temp_dp2 = max(temp_dp2, dp1 + 1)
            if nums2[i] >= nums2[i-1]:
                temp_dp2 = max(temp_dp2, dp2 + 1)
            
            # Update the dp states for the next iteration.
            dp1 = temp_dp1
            dp2 = temp_dp2
            
            # The longest subarray might end at any index, so we update the overall 
            # maximum length in each step.
            max_length = max(max_length, dp1, dp2)
            
        return max_length