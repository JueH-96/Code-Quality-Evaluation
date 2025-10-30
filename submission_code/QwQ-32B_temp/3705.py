from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        n = len(nums)
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            unique = set(sub)
            for num in unique:
                count[num] += 1
        candidates = [num for num, cnt in count.items() if cnt == 1]
        return max(candidates) if candidates else -1