from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        max_len = 1
        prev_num1 = nums1[0]
        prev_num2 = nums2[0]
        prev_len1 = 1
        prev_len2 = 1
        
        max_len = max(prev_len1, prev_len2)
        
        for i in range(1, n):
            curr_num1 = nums1[i]
            curr_num2 = nums2[i]
            
            # Calculate current lengths for choosing nums1[i]
            curr_len1 = 1
            if curr_num1 >= prev_num1:
                curr_len1 = max(curr_len1, prev_len1 + 1)
            if curr_num1 >= prev_num2:
                curr_len1 = max(curr_len1, prev_len2 + 1)
            
            # Calculate current lengths for choosing nums2[i]
            curr_len2 = 1
            if curr_num2 >= prev_num1:
                curr_len2 = max(curr_len2, prev_len1 + 1)
            if curr_num2 >= prev_num2:
                curr_len2 = max(curr_len2, prev_len2 + 1)
            
            # Update the maximum length found so far
            current_max = max(curr_len1, curr_len2)
            if current_max > max_len:
                max_len = current_max
            
            # Prepare for next iteration
            prev_num1, prev_num2 = curr_num1, curr_num2
            prev_len1, prev_len2 = curr_len1, curr_len2
        
        return max_len