import math
from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        """
        Calculates the minimum equal sum achievable by replacing zeros in two arrays
        with strictly positive integers.

        Args:
            nums1: The first list of integers (non-negative).
            nums2: The second list of integers (non-negative).

        Returns:
            The minimum equal sum, or -1 if it's impossible.
        """

        # Calculate the sum of non-zero elements (s1) and the count of zeros (z1) for nums1
        s1 = 0
        z1 = 0
        for num in nums1:
            if num == 0:
                # Each zero must be replaced by a strictly positive integer (>= 1)
                z1 += 1
            else:
                s1 += num

        # Calculate the sum of non-zero elements (s2) and the count of zeros (z2) for nums2
        s2 = 0
        z2 = 0
        for num in nums2:
            if num == 0:
                # Each zero must be replaced by a strictly positive integer (>= 1)
                z2 += 1
            else:
                s2 += num

        # Determine the minimum possible sum for each array after replacing zeros.
        # If an array contains zeros (e.g., z1 > 0), each zero must be replaced by at least 1.
        # Therefore, the minimum sum achievable for nums1 is s1 + z1 * 1 = s1 + z1.
        # If an array has no zeros (e.g., z1 == 0), its sum is fixed at the sum of its elements (s1).
        # Python's integers handle arbitrary size, so potential overflow is not an issue here.
        potential_sum1 = s1 + z1 if z1 > 0 else s1
        potential_sum2 = s2 + z2 if z2 > 0 else s2

        # --- Check for Impossibility ---
        # These checks handle cases where one array has a fixed sum (no zeros),
        # and this fixed sum is inherently smaller than the minimum possible sum
        # achievable by the other array (which might contain zeros).

        # Case 1: nums1 has no zeros (z1 == 0).
        # The sum of nums1 is fixed at s1 (potential_sum1).
        # If this fixed sum s1 is less than the minimum possible sum of nums2 (potential_sum2),
        # then it's impossible to make the sums equal, because the sum of nums2 must be at least potential_sum2.
        if z1 == 0 and potential_sum1 < potential_sum2:
            return -1

        # Case 2: nums2 has no zeros (z2 == 0).
        # The sum of nums2 is fixed at s2 (potential_sum2).
        # If this fixed sum s2 is less than the minimum possible sum of nums1 (potential_sum1),
        # then it's impossible to make the sums equal, because the sum of nums1 must be at least potential_sum1.
        if z2 == 0 and potential_sum2 < potential_sum1:
            return -1

        # --- Calculate Minimum Equal Sum ---
        # If neither of the impossibility conditions above is met, it is always possible
        # to make the sums equal.
        # Let the target equal sum be S. We need S >= potential_sum1 and S >= potential_sum2.
        # The minimum value for S is therefore max(potential_sum1, potential_sum2).
        #
        # Why is this sum achievable?
        # - If both z1 > 0 and z2 > 0: We can always adjust the positive integers replacing the zeros
        #   (making some potentially larger than 1) to make both sums equal to max(potential_sum1, potential_sum2).
        #   For example, to make sum1 = max(potential_sum1, potential_sum2), we need the sum of replacements
        #   for zeros in nums1 to be max(potential_sum1, potential_sum2) - s1.
        #   Since max(potential_sum1, potential_sum2) >= potential_sum1 = s1 + z1,
        #   we have max(potential_sum1, potential_sum2) - s1 >= z1.
        #   This means the required sum of replacements is at least z1, which is possible since we have z1 zeros
        #   and each replacement must be at least 1. A similar argument holds for nums2.
        #
        # - If z1 == 0 and z2 > 0: The impossibility check ensured potential_sum1 >= potential_sum2.
        #   The target sum S must be potential_sum1 (s1). max(potential_sum1, potential_sum2) correctly returns potential_sum1.
        #   We need to show sum2 can reach s1. The required sum of replacements in nums2 is s1 - s2.
        #   Since potential_sum1 >= potential_sum2 => s1 >= s2 + z2 => s1 - s2 >= z2.
        #   This means the required sum (s1 - s2) is achievable with z2 replacements (each >= 1).
        #
        # - If z1 > 0 and z2 == 0: Symmetric to the previous case. The impossibility check ensures
        #   potential_sum2 >= potential_sum1. The target sum is potential_sum2 (s2).
        #   max(potential_sum1, potential_sum2) returns potential_sum2. This sum is achievable for nums1.
        #
        # - If z1 == 0 and z2 == 0: Both impossibility checks ensure potential_sum1 >= potential_sum2
        #   and potential_sum2 >= potential_sum1, which implies potential_sum1 == potential_sum2 (s1 == s2).
        #   max(potential_sum1, potential_sum2) returns this equal sum.
        #
        return max(potential_sum1, potential_sum2)