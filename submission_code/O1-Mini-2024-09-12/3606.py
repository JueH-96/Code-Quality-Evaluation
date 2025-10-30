from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        def digit_sum(n):
            return sum(int(d) for d in str(n))
        replaced = [digit_sum(num) for num in nums]
        return min(replaced)