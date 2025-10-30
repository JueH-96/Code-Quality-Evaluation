from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_changes = n
        for x in range(k + 1):
            current_changes = 0
            for i in range(n // 2):
                if abs(nums[i] - nums[n - 1 - i]) != x:
                    current_changes += 1
            min_changes = min(min_changes, current_changes)
        return min_changes