from collections import defaultdict
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = [(nums[i], nums[n - i - 1]) for i in range(n // 2)]
        
        # Count the frequency of each pair difference
        diff_count = defaultdict(int)
        for a, b in pairs:
            for x in range(k + 1):
                diff_count[abs(a - x)] += 1
                diff_count[abs(b - x)] += 1
        
        # Find the most frequent difference
        max_diff_count = max(diff_count.values())
        
        # Calculate the minimum number of changes
        min_changes = n // 2 - max_diff_count
        
        return min_changes