import collections
from typing import List

class Solution:
    """
    Finds the largest potential outlier in an array based on the problem description.
    The array contains n elements: n-2 special numbers, one element equal to the
    sum of the special numbers, and one outlier element.
    """
    def getLargestOutlier(self, nums: List[int]) -> int:
        """
        Calculates the largest potential outlier value in the given array 'nums'.

        The array 'nums' contains n elements, where n-2 elements are special numbers,
        one element is the sum of these special numbers (S), and one element is an outlier (O).
        This function identifies all possible valid outliers and returns the largest one.

        The logic leverages the property that the total sum of the array elements equals 2*S + O.
        It iterates through unique values 'a' in 'nums', assumes S = a, calculates the
        corresponding potential O, and validates this pair using frequency counts.

        Args:
          nums: A list of integers following the problem's constraints (length >= 3,
                contains n-2 special numbers, 1 sum, 1 outlier).

        Returns:
          The largest integer value among all possible outliers. The problem guarantees
          that at least one such outlier exists.
        """
        n = len(nums)
        
        # Calculate the sum of all elements in nums.
        # Python's sum() handles potential large sums due to arbitrary precision integers.
        total_sum = sum(nums)

        # Count the occurrences of each number in nums using collections.Counter.
        # This provides an efficient way (O(1) average time) to check for the existence
        # and frequency of numbers.
        counts = collections.Counter(nums)

        # Initialize the variable to store the maximum outlier found so far.
        # We use None initially. Since the problem guarantees at least one valid outlier exists,
        # this variable will be updated at least once with a valid integer value.
        max_outlier = None

        # Iterate through each unique number 'a' present in the input list 'nums'.
        # Each unique number is a potential candidate for the 'sum' element (S).
        for a in counts:
            # Let s_val be the candidate value for the sum element S.
            s_val = a
            
            # Calculate the value the outlier 'O' must have if s_val is indeed S.
            # This is derived from the relationship: total_sum = sum_of_special + S + O
            # Since sum_of_special = S, we have total_sum = S + S + O = 2*S + O.
            # Rearranging gives: O = total_sum - 2*S.
            o_val = total_sum - 2 * s_val

            # Check if this calculated potential outlier value 'o_val' is present in the original 'nums' array.
            # The 'counts' dictionary allows this check in O(1) average time.
            if o_val in counts:
                # If 'o_val' exists in 'nums', we have found a potential pair (S=s_val, O=o_val).
                # Now, we need to verify if this assignment is valid based on the counts of these numbers
                # in the original array, ensuring S and O can correspond to elements at distinct indices.
                
                is_valid_scenario = False
                # Scenario 1: The sum candidate (s_val) and the outlier candidate (o_val) have the same value.
                if s_val == o_val:
                    # For this to be valid, the value must appear at least twice in 'nums'.
                    # This allows us to pick one instance as S and another instance (at a different index) as O.
                    if counts[s_val] >= 2:
                        is_valid_scenario = True
                # Scenario 2: The sum candidate (s_val) and the outlier candidate (o_val) have different values.
                else:
                    # If s_val and o_val are different, their mere presence in 'nums' is sufficient
                    # to guarantee they can correspond to distinct elements/indices.
                    # We know 's_val' exists because we are iterating through the keys of 'counts'.
                    # We know 'o_val' exists because it passed the 'o_val in counts' check.
                    is_valid_scenario = True

                # If the current assignment (S=s_val, O=o_val) constitutes a valid scenario,
                # then 'o_val' is a potential outlier according to the problem definition.
                if is_valid_scenario:
                    # Compare this potential outlier 'o_val' with the maximum outlier found so far.
                    # Update 'max_outlier' if this is the first valid outlier found (max_outlier is None)
                    # or if 'o_val' is greater than the current 'max_outlier'.
                    if max_outlier is None or o_val > max_outlier:
                        max_outlier = o_val
                        
        # The problem statement guarantees that at least one potential outlier exists in the input.
        # Therefore, 'max_outlier' will have been assigned a valid integer value after the loop finishes.
        # We return this value.
        return max_outlier