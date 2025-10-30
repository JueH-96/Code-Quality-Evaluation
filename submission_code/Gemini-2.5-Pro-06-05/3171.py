from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the minimum equal sum of two arrays after replacing zeros.
        """
        s1, zeros1 = 0, 0
        for num in nums1:
            if num == 0:
                zeros1 += 1
            # The sum of non-zero elements is calculated. Adding 0 doesn't affect the sum.
            s1 += num

        s2, zeros2 = 0, 0
        for num in nums2:
            if num == 0:
                zeros2 += 1
            s2 += num
            
        # The minimum sum for an array is achieved by replacing all 0s with 1s.
        # This is the sum of non-zero elements plus the number of zeros.
        min_sum1 = s1 + zeros1
        min_sum2 = s2 + zeros2
        
        # If an array has no zeros, its sum is fixed. If this fixed sum is less than
        # the minimum possible sum of the other array, it's impossible to make them equal.
        
        # Case 1: nums1 has no zeros, and its fixed sum is less than nums2's minimum.
        if zeros1 == 0 and min_sum1 < min_sum2:
            return -1
            
        # Case 2: nums2 has no zeros, and its fixed sum is less than nums1's minimum.
        if zeros2 == 0 and min_sum2 < min_sum1:
            return -1
            
        # In all other scenarios where a solution is possible, the minimum equal sum
        # is the maximum of their individual minimum possible sums.
        # - If both arrays have zeros, they can achieve any sum >= their minimum.
        #   The smallest common sum is the maximum of these minimums.
        # - If one array has no zeros (e.g., nums1), the checks above ensure
        #   min_sum1 >= min_sum2. So, max(min_sum1, min_sum2) correctly yields
        #   min_sum1 (the required fixed sum).
        return max(min_sum1, min_sum2)