from typing import List
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = defaultdict(int)
        
        for i in range(n // 2):
            diff = abs(nums[i] - nums[n - i - 1])
            count[diff] += 1
        
        max_count = max(count.values(), default=0)
        
        if max_count == n // 2:
            return 0
        
        min_changes = n // 2
        
        for diff in count:
            changes = 0
            for i in range(n // 2):
                if abs(nums[i] - nums[n - i - 1]) != diff:
                    if nums[i] > nums[n - i - 1]:
                        if nums[i] - diff > k:
                            changes += 1
                        else:
                            nums[i] = nums[n - i - 1] + diff
                    else:
                        if nums[n - i - 1] - diff > k:
                            changes += 1
                        else:
                            nums[n - i - 1] = nums[i] + diff
            min_changes = min(min_changes, changes)
        
        return min_changes