from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # For any number, if it is divisible by 3 then no operation is required.
        # If not, whether the remainder is 1 or 2, the minimum operation is 1
        # (subtract 1 if remainder is 1, or add 1 if remainder is 2).
        # Thus, we only need to count how many elements are not divisible by 3.
        return sum(1 for num in nums if num % 3 != 0)