from typing import List

class Solution:
  def minSum(self, nums1: List[int], nums2: List[int]) -> int:
    s1_val = 0
    z1_count = 0
    # Calculate sum of elements and count of zeros for nums1
    for x in nums1:
        s1_val += x # sum() behavior: sums all elements, 0s contribute 0.
        if x == 0:
            z1_count += 1

    s2_val = 0
    z2_count = 0
    # Calculate sum of elements and count of zeros for nums2
    for x in nums2:
        s2_val += x
        if x == 0:
            z2_count += 1
    
    # min_sum_for_nums1 is the sum of nums1 if all its 0s are replaced by 1s.
    # If nums1 has no 0s (z1_count == 0), then its sum is fixed at s1_val.
    # In this case, min_sum_for_nums1 correctly becomes s1_val because z1_count is 0.
    min_sum_for_nums1 = s1_val + z1_count
    
    # Similarly for nums2
    min_sum_for_nums2 = s2_val + z2_count

    # Impossibility Check 1:
    # If nums1 has no zeros, its sum is fixed (s1_val, which equals min_sum_for_nums1).
    # If this fixed sum is less than the minimum sum nums2 can achieve (min_sum_for_nums2),
    # then it's impossible to make their sums equal.
    if z1_count == 0: # Sum of nums1 is fixed
        if min_sum_for_nums1 < min_sum_for_nums2:
            return -1
    
    # Impossibility Check 2:
    # Symmetrically, if nums2 has no zeros, its sum is fixed (s2_val, which equals min_sum_for_nums2).
    # If this fixed sum is less than the minimum sum nums1 can achieve (min_sum_for_nums1),
    # then it's impossible.
    if z2_count == 0: # Sum of nums2 is fixed
        if min_sum_for_nums2 < min_sum_for_nums1:
            return -1
            
    # If neither of the above impossibility conditions is met, it is possible.
    # The minimum equal sum will be the maximum of the minimum possible sums of the two arrays.
    # This is because:
    # - If an array has zeros, it can achieve any sum greater than or equal to its minimum sum.
    # - If an array has no zeros, its sum is fixed. The impossibility checks ensure that this fixed sum
    #   is compatible with the other array's sum range. Specifically, if nums1 has no zeros, then
    #   min_sum_for_nums1 >= min_sum_for_nums2, so max() will yield min_sum_for_nums1.
    #   This sum is achievable by nums1 (it's its fixed sum) and by nums2 (as min_sum_for_nums1 >= min_sum_for_nums2).
    #   A similar argument applies if nums2 has no zeros.
    # - If both arrays have zeros, they can both achieve any sum greater than or equal to their respective minimums.
    #   Thus, they can both achieve the target sum max(min_sum_for_nums1, min_sum_for_nums2).
    return max(min_sum_for_nums1, min_sum_for_nums2)