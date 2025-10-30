import math
from functools import reduce
from typing import List

# Starter code structure provided by the user
class Solution:
    """
    Calculates the maximum factor score of an array nums after removing at most one element.
    The factor score is defined as the product of the LCM and GCD of all elements of the array.
    """
    def maxScore(self, nums: List[int]) -> int:
        """
        Finds the maximum factor score achievable by removing at most one element from nums.

        Args:
            nums: A list of integers. Constraints: 1 <= nums.length <= 100, 1 <= nums[i] <= 30.

        Returns:
            The maximum factor score.
        """

        # --- Helper Functions ---

        def _calculate_gcd(arr: List[int]) -> int:
            """
            Calculates the Greatest Common Divisor (GCD) of a non-empty list of positive integers.
            Assumes the input list 'arr' is not empty and contains only positive integers (as per constraints).
            """
            # Base case: If the list contains only one element, its GCD is the element itself.
            if len(arr) == 1:
                return arr[0]
            # Use functools.reduce combined with math.gcd to compute the GCD of all elements.
            # math.gcd(a, b) finds the GCD of two numbers.
            # reduce applies this function cumulatively to the elements of the list.
            # Example: reduce(gcd, [a, b, c]) is equivalent to gcd(gcd(a, b), c).
            return reduce(math.gcd, arr)

        def _calculate_lcm(arr: List[int]) -> int:
            """
            Calculates the Least Common Multiple (LCM) of a non-empty list of positive integers.
            Assumes the input list 'arr' is not empty and contains only positive integers.
            """
            # Base case: If the list contains only one element, its LCM is the element itself.
            if len(arr) == 1:
                return arr[0]
            
            # Initialize the LCM result 'l' with the first element of the list.
            l = arr[0]
            # Iterate through the remaining elements of the list starting from the second element.
            for i in range(1, len(arr)):
                current_num = arr[i]
                # Calculate the GCD of the currently accumulated LCM 'l' and the next number 'current_num'.
                # This GCD is needed for the LCM formula. Requires 'import math'.
                current_gcd = math.gcd(l, current_num)
                
                # Update the accumulated LCM 'l' using the formula: lcm(a, b) = (a * b) // gcd(a, b).
                # Python's integers have arbitrary precision, so intermediate multiplication (l * current_num)
                # will not overflow standard integer types.
                # Since constraints state nums[i] >= 1, all numbers are positive, and gcd will be >= 1.
                # Thus, division by zero is not a concern.
                # Perform multiplication first, then integer division.
                l = (l * current_num) // current_gcd
            
            # Return the final calculated LCM of all elements in the list.
            return l

        def _calculate_score(arr: List[int]) -> int:
            """
            Calculates the factor score for a given list 'arr'.
            Factor Score = GCD(arr) * LCM(arr).
            Handles the empty list case as defined in the problem.
            """
            # According to the problem definition, the factor score of an empty array is 0.
            if not arr:
                return 0 
            
            # If the list is not empty, calculate its GCD and LCM using the helper functions.
            g = _calculate_gcd(arr)
            l = _calculate_lcm(arr)
            
            # The factor score is defined as the product of the GCD and LCM.
            return g * l

        # --- Main Logic ---

        # Initialize the variable to store the maximum score found so far.
        max_score = 0
        n = len(nums) # Get the number of elements in the input list.

        # Constraints guarantee 1 <= n <= 100, so the input list 'nums' is never empty.

        # First, calculate the score for the original list without removing any elements.
        # This provides the baseline score.
        max_score = _calculate_score(nums)

        # Next, iterate through each element of the list. In each iteration,
        # consider the scenario where the element at the current index 'i' is removed.
        for i in range(n):
            # Create a temporary list 'temp_nums' by excluding the element nums[i].
            # This is done using list slicing: concatenating the part before index 'i'
            # with the part after index 'i'.
            temp_nums = nums[:i] + nums[i+1:]
            
            # Calculate the factor score for this temporary list (which might be empty if n=1).
            # The _calculate_score function correctly handles the empty list case, returning 0.
            current_score = _calculate_score(temp_nums)
            
            # Update 'max_score' if the score calculated for the list with one element removed
            # ('current_score') is greater than the current maximum score found so far.
            max_score = max(max_score, current_score)
            
        # After checking the score of the original list and the scores of all possible lists
        # formed by removing one element, 'max_score' holds the overall maximum factor score.
        return max_score