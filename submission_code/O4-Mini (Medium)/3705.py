from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Count how many windows of size k each number appears in
        count_map = defaultdict(int)
        n = len(nums)
        
        # Slide a window of size k over nums
        for i in range(n - k + 1):
            window = nums[i:i+k]
            # Use a set to count each number at most once per window
            for x in set(window):
                count_map[x] += 1
        
        # Find the largest number that appears in exactly one window
        ans = -1
        for x, cnt in count_map.items():
            if cnt == 1:
                ans = max(ans, x)
        
        return ans