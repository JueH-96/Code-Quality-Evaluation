from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # prev_dp1: max length of a non-decreasing subarray ending at (i-1)
        #           where nums3[i-1] was chosen as nums1[i-1].
        # prev_dp2: max length of a non-decreasing subarray ending at (i-1)
        #           where nums3[i-1] was chosen as nums2[i-1].
        
        # For the first element (index 0), any choice forms a non-decreasing subarray of length 1.
        prev_dp1 = 1 
        prev_dp2 = 1
        
        # The overall maximum length found so far.
        # Since n >= 1 (per constraints), at least one element exists, 
        # so the minimum possible non-decreasing subarray length is 1.
        max_len = 1 

        # Iterate from the second element (index 1) to the end.
        for i in range(1, n):
            # current_dp1: max length ending at 'i' choosing nums1[i]
            # current_dp2: max length ending at 'i' choosing nums2[i]
            
            # Initialize current_dp values to 1. This handles cases where
            # nums1[i] or nums2[i] cannot extend any previous subarray,
            # effectively starting a new non-decreasing subarray of length 1.
            current_dp1 = 1 
            current_dp2 = 1

            # Option for nums3[i] = nums1[i]:
            # Check if nums1[i] can extend a subarray ending with nums1[i-1]
            if nums1[i] >= nums1[i-1]:
                current_dp1 = max(current_dp1, prev_dp1 + 1)
            # Check if nums1[i] can extend a subarray ending with nums2[i-1]
            if nums1[i] >= nums2[i-1]:
                current_dp1 = max(current_dp1, prev_dp2 + 1)
            
            # Option for nums3[i] = nums2[i]:
            # Check if nums2[i] can extend a subarray ending with nums1[i-1]
            if nums2[i] >= nums1[i-1]:
                current_dp2 = max(current_dp2, prev_dp1 + 1)
            # Check if nums2[i] can extend a subarray ending with nums2[i-1]
            if nums2[i] >= nums2[i-1]:
                current_dp2 = max(current_dp2, prev_dp2 + 1)
            
            # Update the overall maximum length found so far
            max_len = max(max_len, current_dp1, current_dp2)
            
            # Prepare for the next iteration: current values become previous values
            prev_dp1 = current_dp1
            prev_dp2 = current_dp2
            
        return max_len