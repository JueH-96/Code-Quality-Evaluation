from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        """
        Calculates the maximum sum of a subarray with unique elements after potentially deleting elements from nums.

        The problem statement allows deleting elements to form a modified array (subsequence B) 
        and then finding a subarray (C) of B with unique elements and maximum sum.
        We proved that this is equivalent to finding a subsequence of the original nums 
        that has unique elements and the maximum possible sum.

        Algorithm:
        1. Find all unique numbers present in the input array `nums`.
        2. Sum all the unique positive numbers. Let this be `positive_sum`.
        3. If there are any unique positive numbers (`positive_sum > 0`), then the maximum possible sum 
           is `positive_sum`. This is because including any non-positive number (0 or negative) 
           would either not increase the sum or decrease it. The subsequence containing just the 
           unique positive numbers achieves this sum.
        4. If there are no unique positive numbers (`positive_sum == 0`), it means all unique numbers 
           in `nums` are less than or equal to 0.
           a. If 0 is present among the unique numbers, the maximum sum is 0. This can be achieved 
              by the subsequence `[0]`. Any other subsequence will have a sum less than or equal to 0.
           b. If 0 is not present, it means all unique numbers are strictly negative. Since the original 
              array `nums` is non-empty, the set of unique elements is also non-empty. The problem 
              implies the final selection must be non-empty (cannot delete all elements). To maximize 
              the sum when all available unique numbers are negative, we must choose at least one. 
              The best choice is the single largest negative number (the one closest to zero). 
              The subsequence `[max(unique_negative_numbers)]` achieves this maximum negative sum.

        Args:
          nums: The input list of integers. Constraints: 1 <= nums.length <= 100, -100 <= nums[i] <= 100.

        Returns:
          The maximum possible sum according to the problem rules.
        """
        
        # Step 1: Find the set of unique elements in the input array.
        # Using a set automatically handles duplicates and provides efficient lookups.
        distinct_elements = set(nums)
        
        # Step 2: Calculate the sum of all unique positive elements.
        positive_sum = 0
        # Use a flag to efficiently track if at least one positive number exists.
        has_positive = False
        for x in distinct_elements:
            if x > 0:
                positive_sum += x
                has_positive = True
                
        # Step 3: Check if any positive unique elements were found.
        if has_positive:
            # If yes, the maximum sum is the sum of all unique positive elements.
            return positive_sum
        # Step 4: Handle the case where all unique elements are non-positive (<= 0).
        else:
            # Subcase 4a: Check if 0 is among the unique elements.
            if 0 in distinct_elements:
                # If 0 is present, the maximum achievable sum is 0.
                return 0
            # Subcase 4b: All unique elements are strictly negative.
            else:
                # The input `nums` is guaranteed non-empty, so `distinct_elements` is non-empty.
                # Since all unique elements are negative, the maximum sum is the largest negative element.
                return max(distinct_elements)