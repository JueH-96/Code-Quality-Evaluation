from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1_orig = 0
        zeros1 = 0
        for x in nums1:
            if x == 0:
                zeros1 += 1
            else:
                sum1_orig += x
        
        sum2_orig = 0
        zeros2 = 0
        for x in nums2:
            if x == 0:
                zeros2 += 1
            else:
                sum2_orig += x
                
        # Calculate the minimum possible sum for each array.
        # This is achieved by replacing each 0 with the smallest strictly positive integer, which is 1.
        min_sum1_possible = sum1_orig + zeros1
        min_sum2_possible = sum2_orig + zeros2
        
        # Case 1: Both arrays have zeros.
        # If both arrays contain zeros, we can adjust the sum of either array upwards
        # from its minimum possible value by assigning larger values to its zeros.
        # The minimum common sum will be the maximum of their individual minimum possible sums.
        # For example, if min_sum1_possible < min_sum2_possible, we can make nums1's sum equal
        # to min_sum2_possible by replacing its zeros such that their sum is (min_sum2_possible - sum1_orig).
        # Since (min_sum2_possible - sum1_orig) >= min_sum1_possible - sum1_orig = zeros1,
        # this sum can be distributed among zeros1 positive integers.
        if zeros1 > 0 and zeros2 > 0:
            return max(min_sum1_possible, min_sum2_possible)
        
        # Case 2: Only nums1 has zeros (nums2 has no zeros, so its sum is fixed).
        # The target equal sum must be sum2_orig.
        # We need to check if nums1 can achieve this sum.
        # The smallest sum nums1 can achieve is min_sum1_possible.
        # If sum2_orig is less than min_sum1_possible, it's impossible for nums1 to reach sum2_orig,
        # because nums1's sum would be too low even with all zeros replaced by 1s.
        # Otherwise (if sum2_orig >= min_sum1_possible), nums1 can indeed achieve sum2_orig
        # by distributing the required value (sum2_orig - sum1_orig) among its zeros.
        elif zeros1 > 0 and zeros2 == 0:
            if sum2_orig < min_sum1_possible:
                return -1
            else:
                return sum2_orig 
        
        # Case 3: Only nums2 has zeros (nums1 has no zeros, so its sum is fixed).
        # This is symmetric to Case 2. The target equal sum must be sum1_orig.
        # We check if nums2 can achieve this sum.
        elif zeros1 == 0 and zeros2 > 0:
            if sum1_orig < min_sum2_possible:
                return -1
            else:
                return sum1_orig
                
        # Case 4: Neither array has zeros (both sums are fixed).
        # For the sums to be equal, their original sums must be identical.
        # If they are equal, that's the minimum equal sum. Otherwise, it's impossible.
        else: # zeros1 == 0 and zeros2 == 0
            if sum1_orig == sum2_orig:
                return sum1_orig
            else:
                return -1