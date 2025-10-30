from typing import List
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')
        for x in range(k + 1):
            changes = 0
            counts = defaultdict(int)
            for i in range(n // 2):
                if abs(nums[i] - nums[n - 1 - i]) != x:
                    if nums[i] <= k and nums[n - 1 - i] <= k:
                        if nums[i] - nums[n - 1 - i] == x or nums[n - 1 - i] - nums[i] == x:
                            changes += 1
                        else:
                            changes += 2
                    elif nums[i] <= k:
                        if nums[i] + x <= k or nums[i] - x >= 0:
                            changes += 1
                        else:
                            changes += 2
                    elif nums[n - 1 - i] <= k:
                        if nums[n - 1 - i] + x <= k or nums[n - 1 - i] - x >= 0:
                            changes += 1
                        else:
                            changes += 2
                    else:
                        changes += 2
            ans = min(ans, changes)
        return ans