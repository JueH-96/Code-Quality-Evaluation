from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)

        # Constraints state 3 <= nums.length.
        # So, n is guaranteed to be at least 3.
        
        # A subarray of length 3 is formed by (nums[i], nums[i+1], nums[i+2]).
        # The index i can range from 0 up to n-3 (inclusive).
        # In Python, range(n-2) generates integers from 0 up to n-3.
        # For example, if n=3, range(1) gives i=0. The subarray is (nums[0], nums[1], nums[2]).
        # If n=5, range(3) gives i=0, 1, 2. The last subarray for i=2 is (nums[2], nums[3], nums[4]).
        
        for i in range(n - 2):
            first_element = nums[i]
            middle_element = nums[i+1]
            last_element = nums[i+2]
            
            # The condition specified is:
            #   sum_of_ends == half_of_middle
            #   first_element + last_element == middle_element / 2.0
            #
            # To avoid floating-point arithmetic and potential precision issues,
            # we can multiply both sides of the equation by 2:
            #   2 * (first_element + last_element) == middle_element
            #
            # This transformation is robust:
            # - If middle_element is odd, middle_element / 2.0 is a non-integer (e.g., X.5).
            #   Since (first_element + last_element) is an integer, it can never equal X.5.
            #   The original condition is false.
            #   In the transformed condition, 2 * (first_element + last_element) is even,
            #   while middle_element (if odd) is odd. 'even == odd' is false.
            #   Both correctly yield false.
            # - If middle_element is even, the transformation is a standard algebraic manipulation.
            
            if 2 * (first_element + last_element) == middle_element:
                count += 1
                
        return count