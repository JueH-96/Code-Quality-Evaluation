from typing import List

class Solution:
    def sum_digits(self, n: int) -> int:
        """Calculates the sum of digits of a non-negative integer."""
        total = 0
        # Since constraints say 1 <= nums[i], n will be positive.
        while n > 0:
            total += n % 10
            n //= 10
        return total

    def minElement(self, nums: List[int]) -> int:
        """
        Replaces each element in nums with the sum of its digits
        and returns the minimum element in the modified list.
        """
        # Replace each element with the sum of its digits
        for i in range(len(nums)):
            nums[i] = self.sum_digits(nums[i])

        # Find the minimum element in the modified list
        return min(nums)