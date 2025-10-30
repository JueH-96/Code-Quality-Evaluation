from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        sums = [sum(int(d) for d in str(num)) for num in nums]
        return min(sums)