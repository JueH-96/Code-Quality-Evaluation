import math
from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        """
        Replaces each element in nums with the sum of its digits,
        then returns the minimum element in the modified list.

        Args:
            nums: A list of integers.

        Returns:
            The minimum element after replacing each number with the sum of its digits.
        """

        # Helper function to calculate the sum of digits for a single number
        def sum_digits(n: int) -> int:
            """Calculates the sum of the digits of a non-negative integer."""
            # Base case: if the number is 0, the sum is 0.
            # Although constraints state nums[i] >= 1, handling 0 is good practice.
            if n == 0:
                return 0
                
            current_sum = 0
            # Constraints guarantee n > 0
            while n > 0:
                # Add the last digit (n % 10) to the sum
                current_sum += n % 10
                # Remove the last digit by integer division (n // 10)
                n //= 10
            return current_sum

        # Initialize minimum_sum to positive infinity to ensure the first calculated sum becomes the minimum
        minimum_sum = float('inf') 

        # Iterate through each number in the input list
        for num in nums:
            # Calculate the sum of digits for the current number
            current_digit_sum = sum_digits(num)
            # Update the overall minimum if the current sum is smaller
            if current_digit_sum < minimum_sum:
                minimum_sum = current_digit_sum
                
        # Return the minimum sum found
        return minimum_sum

# Example Usage (for testing outside the class structure if needed)
# solver = Solution()
# nums1 = [10,12,13,14]
# print(f"Input: {nums1}, Output: {solver.minElement(nums1)}") # Expected: 1

# nums2 = [1,2,3,4]
# print(f"Input: {nums2}, Output: {solver.minElement(nums2)}") # Expected: 1

# nums3 = [999,19,199]
# print(f"Input: {nums3}, Output: {solver.minElement(nums3)}") # Expected: 10

# Alternative implementation using list comprehension and min()
class Solution_concise:
     def minElement(self, nums: List[int]) -> int:
        
        def sum_digits(n: int) -> int:
            s = 0
            while n > 0:
                s += n % 10
                n //= 10
            return s

        # Create a new list containing the sum of digits for each number
        digit_sums = [sum_digits(num) for num in nums]
        
        # Find and return the minimum element in the new list
        # min() is safe because constraints guarantee nums is not empty (1 <= nums.length)
        return min(digit_sums)