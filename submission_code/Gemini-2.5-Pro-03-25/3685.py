import math
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Counts the number of subarrays of length 3 where the sum of the first 
        and third elements equals half the second element.

        Args:
            nums: A list of integers.

        Returns:
            The count of such subarrays.
        """
        count = 0
        n = len(nums)

        # A subarray of length 3 needs at least 3 elements in the original array.
        # The constraint nums.length >= 3 ensures this is possible.
        
        # We need to iterate through all possible starting indices `i` such that
        # a subarray of length 3 can be formed starting at `i`.
        # The subarray starting at index `i` is nums[i], nums[i+1], nums[i+2].
        # The last possible index for `i` is `n - 3` so that `i + 2` is `n - 1`,
        # which is the last valid index of the array.
        # Therefore, the loop should go from i = 0 to n - 3 (inclusive).
        # range(n - 2) generates numbers from 0 up to n - 3.
        for i in range(n - 2):
            # Get the elements of the current subarray of length 3
            first_num = nums[i]
            second_num = nums[i+1]
            third_num = nums[i+2]

            # Check the condition: sum of first and third equals half the second.
            # To avoid potential floating point issues with division, we can rewrite
            # first_num + third_num == second_num / 2
            # as
            # 2 * (first_num + third_num) == second_num
            # This works correctly even if second_num is odd, because if second_num
            # is odd, second_num / 2 would not be an integer, and thus could never
            # equal the integer sum first_num + third_num. The multiplied version
            # handles this correctly: 2 * integer_sum will always be even, while
            # an odd second_num cannot be equal to an even number.
            if 2 * (first_num + third_num) == second_num:
                count += 1
                
        return count