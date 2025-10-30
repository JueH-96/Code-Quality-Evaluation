from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        freq = defaultdict(int)
        n = len(nums)
        for s in range(0, n - k + 1):
            sub = nums[s:s + k]
            unique_sub = set(sub)
            for x in unique_sub:
                freq[x] += 1
        candidates = [x for x, cnt in freq.items() if cnt == 1]
        return max(candidates) if candidates else -1