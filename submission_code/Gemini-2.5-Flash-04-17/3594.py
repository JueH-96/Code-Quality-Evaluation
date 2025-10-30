from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        """
        Finds the largest potential outlier in an array where n-2 elements
        sum up to one element, and one element is the outlier.

        Args:
            nums: The input list of integers.

        Returns:
            The largest potential outlier.
        """
        # Calculate the total sum of all elements in the array.
        # This is used in the equation TotalSum = 2 * SumElement + Outlier
        total_sum = sum(nums)

        # Count the frequency of each number in the array.
        # This helps check if required potential sum/outlier values exist
        # with sufficient counts.
        counts = Counter(nums)

        # Set to store all potential outlier values found. Using a set
        # automatically handles unique values.
        potential_outliers = set()

        # Iterate through each unique value present in the array.
        # Each unique value 'val' is a candidate for either the SumElement
        # or the Outlier element from the pair of non-special numbers.
        for val in counts.keys():
            # Let 'val' be one of the two non-special numbers.
            # Let the other non-special number be 'other_val'.
            # The sum of special numbers is S. The two non-special numbers are S and O.
            # TotalSum = S + S + O = 2*S + O

            # Scenario 1: 'val' is the sum of the special numbers (S = val).
            # The equation is TotalSum = 2 * val + O.
            # The potential outlier value O must be TotalSum - 2 * val.
            v_outlier_candidate = total_sum - 2 * val

            # We need to check if this pair (val as sum, v_outlier_candidate as outlier)
            # can actually be removed from the original counts.
            # We need to remove one instance of 'val' and one instance of 'v_outlier_candidate'.
            if val == v_outlier_candidate:
                # If the sum value and the outlier value are the same ('val'),
                # we need to remove two instances of 'val'. This is possible
                # if the count of 'val' in the original array is at least 2.
                if counts[val] >= 2:
                    potential_outliers.add(v_outlier_candidate) # Add the outlier value
            else:
                # If the sum value ('val') and the outlier value ('v_outlier_candidate')
                # are different, we need to remove one instance of 'val' and
                # one instance of 'v_outlier_candidate'. This is possible if
                # counts[val] >= 1 (which is true since 'val' is a key in counts)
                # AND counts[v_outlier_candidate] >= 1.
                if counts.get(v_outlier_candidate, 0) >= 1:
                    potential_outliers.add(v_outlier_candidate) # Add the outlier value


            # Scenario 2: 'val' is the outlier (O = val).
            # The equation is TotalSum = 2 * S + val.
            # This implies 2 * S = TotalSum - val.
            # The potential sum value S must be (TotalSum - val) / 2.

            required_sum_num = total_sum - val

            # For S to be an integer value, (TotalSum - val) must be even.
            if required_sum_num % 2 == 0:
                v_sum_candidate = required_sum_num // 2

                # We need to check if this pair (v_sum_candidate as sum, val as outlier)
                # can actually be removed from the original counts.
                # We need to remove one instance of 'v_sum_candidate' and one instance of 'val'.
                if val == v_sum_candidate:
                     # If the outlier value ('val') and the sum value ('v_sum_candidate')
                     # are the same, we need to remove two instances of 'val'.
                     # This is possible if counts[val] >= 2.
                     # This specific case (val == v_outlier_candidate in Scen 1 AND
                     # val == v_sum_candidate in Scen 2) happens when 3*val == total_sum.
                     # The check 'if counts[val] >= 2' in Scenario 1 for val == v_outlier_candidate
                     # correctly handles this scenario, adding 'val' as a potential outlier.
                     # No explicit action is needed here if the check in Scenario 1 was done.
                     pass # The logic structure ensures this is covered

                else:
                    # If the outlier value ('val') and the sum value ('v_sum_candidate')
                    # are different, we need to remove one instance of 'val' and
                    # one instance of 'v_sum_candidate'. This is possible if
                    # counts[val] >= 1 (true) and counts[v_sum_candidate] >= 1.
                    if counts.get(v_sum_candidate, 0) >= 1:
                        potential_outliers.add(val) # Add the outlier value (which is val)

        # The problem guarantees that at least one potential outlier exists,
        # so the set potential_outliers will not be empty.
        # Return the maximum value among all potential outliers found.
        return max(potential_outliers)