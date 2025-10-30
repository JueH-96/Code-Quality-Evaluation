from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0  # Though constraints ensure n >= 1, handling edge case
        
        # Initialize lengths for index 0
        len_end_with_nums1 = 1
        len_end_with_nums2 = 1
        max_length = 1
        
        # Iterate from index 1 to n-1
        for i in range(1, n):
            # Compute new length if ending with nums1[i]
            cand_new_nums1 = 0
            if nums1[i-1] <= nums1[i]:
                cand_new_nums1 = max(cand_new_nums1, len_end_with_nums1)
            if nums2[i-1] <= nums1[i]:
                cand_new_nums1 = max(cand_new_nums1, len_end_with_nums2)
            new_len_nums1 = 1 + cand_new_nums1
            
            # Compute new length if ending with nums2[i]
            cand_new_nums2 = 0
            if nums1[i-1] <= nums2[i]:
                cand_new_nums2 = max(cand_new_nums2, len_end_with_nums1)
            if nums2[i-1] <= nums2[i]:
                cand_new_nums2 = max(cand_new_nums2, len_end_with_nums2)
            new_len_nums2 = 1 + cand_new_nums2
            
            # Update the maximum length found
            max_length = max(max_length, new_len_nums1, new_len_nums2)
            
            # Update the lengths for the next iteration
            len_end_with_nums1 = new_len_nums1
            len_end_with_nums2 = new_len_nums2
        
        return max_length