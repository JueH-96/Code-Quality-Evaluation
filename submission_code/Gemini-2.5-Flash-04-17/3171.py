from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum of non-zero elements and count of zeros for nums1
        sum1 = 0
        zero_count1 = 0
        for x in nums1:
            if x == 0:
                zero_count1 += 1
            else:
                sum1 += x

        # Calculate sum of non-zero elements and count of zeros for nums2
        sum2 = 0
        zero_count2 = 0
        for x in nums2:
            if x == 0:
                zero_count2 += 1
            else:
                sum2 += x

        # Analyze cases based on whether each array contains zeros
        # When replacing zeros with strictly positive integers (>= 1),
        # an array's sum can be increased by at least the number of zeros it contains.
        # If an array has zeros (count > 0), its sum can be any integer >= base_sum + zero_count.
        # If an array has no zeros (count == 0), its sum is fixed at base_sum.

        # Calculate the minimum possible sum for each array if it has zeros
        # This value represents the lower bound of the sum range when zeros exist.
        # If zero_count is 0, this value doesn't represent a range lower bound
        # in the sense of being able to achieve any sum >= this value.
        # Instead, the sum is fixed.
        s1_min_sum_if_zeros = sum1 + zero_count1
        s2_min_sum_if_zeros = sum2 + zero_count2

        # We are looking for the minimum sum S such that S is achievable by both arrays.
        # Achievable sums for nums1: {sum1} if zero_count1 == 0, else [s1_min_sum_if_zeros, infinity)
        # Achievable sums for nums2: {sum2} if zero_count2 == 0, else [s2_min_sum_if_zeros, infinity)
        # We need to find the minimum value in the intersection of these two sets.

        if zero_count1 == 0 and zero_count2 == 0:
            # Case 1: No zeros in either array. Sums are fixed at sum1 and sum2.
            # Possible only if sums are already equal.
            return sum1 if sum1 == sum2 else -1

        elif zero_count1 > 0 and zero_count2 == 0:
            # Case 2: Zeros in nums1, no zeros in nums2.
            # nums1 can achieve any sum >= s1_min_sum_if_zeros.
            # nums2 must have sum2.
            # Equal sum S must be sum2. This is possible only if sum2 is achievable by nums1.
            # i.e., sum2 is in the range [s1_min_sum_if_zeros, infinity), which means sum2 >= s1_min_sum_if_zeros.
            return sum2 if sum2 >= s1_min_sum_if_zeros else -1

        elif zero_count1 == 0 and zero_count2 > 0:
            # Case 3: No zeros in nums1, zeros in nums2.
            # nums1 must have sum1.
            # nums2 can achieve any sum >= s2_min_sum_if_zeros.
            # Equal sum S must be sum1. This is possible only if sum1 is achievable by nums2.
            # i.e., sum1 is in the range [s2_min_sum_if_zeros, infinity), which means sum1 >= s2_min_sum_if_zeros.
            return sum1 if sum1 >= s2_min_sum_if_zeros else -1

        else: # zero_count1 > 0 and zero_count2 > 0
            # Case 4: Zeros in both arrays.
            # nums1 can achieve any sum >= s1_min_sum_if_zeros.
            # nums2 can achieve any sum >= s2_min_sum_if_zeros.
            # The set of achievable sums for nums1 is [s1_min_sum_if_zeros, infinity).
            # The set of achievable sums for nums2 is [s2_min_sum_if_zeros, infinity).
            # The intersection of these two sets is [max(s1_min_sum_if_zeros, s2_min_sum_if_zeros), infinity).
            # The minimum common sum is the maximum of the two lower bounds.
            return max(s1_min_sum_if_zeros, s2_min_sum_if_zeros)