from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n: int) -> int:
            return sum(int(digit) for digit in str(n))

        # Replace each element in nums with the sum of its digits
        replaced_nums = [sum_of_digits(num) for num in nums]

        # Return the minimum element in the replaced nums
        return min(replaced_nums)