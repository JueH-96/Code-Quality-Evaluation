from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Count of subarrays of size k in which each number appears
        cnt = defaultdict(int)
        
        # Slide over all subarrays of length k
        for i in range(n - k + 1):
            window = set(nums[i:i + k])
            for x in window:
                cnt[x] += 1
        
        # Find the largest x with count == 1
        ans = -1
        for x, c in cnt.items():
            if c == 1 and x > ans:
                ans = x
        
        return ans