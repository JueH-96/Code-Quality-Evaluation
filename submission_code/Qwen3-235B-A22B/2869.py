from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        max_len = 1
        prev_dp1 = 1  # previous dp value for nums1
        prev_dp2 = 1  # previous dp value for nums2
        
        for i in range(1, n):
            current_dp1 = 1
            current_dp2 = 1
            
            # Update current_dp1 (using nums1[i])
            if nums1[i] >= nums1[i-1]:
                current_dp1 = max(current_dp1, prev_dp1 + 1)
            if nums1[i] >= nums2[i-1]:
                current_dp1 = max(current_dp1, prev_dp2 + 1)
            
            # Update current_dp2 (using nums2[i])
            if nums2[i] >= nums1[i-1]:
                current_dp2 = max(current_dp2, prev_dp1 + 1)
            if nums2[i] >= nums2[i-1]:
                current_dp2 = max(current_dp2, prev_dp2 + 1)
            
            # Update the maximum length found
            current_max = max(current_dp1, current_dp2)
            if current_max > max_len:
                max_len = current_max
            
            # Prepare for next iteration
            prev_dp1, prev_dp2 = current_dp1, current_dp2
        
        return max_len