from typing import List
from bisect import bisect_left, bisect_right

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        pos = [i for i, num in enumerate(nums) if num == x]
        return [pos[query - 1] if query <= len(pos) else -1 for query in queries]