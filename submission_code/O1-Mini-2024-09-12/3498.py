from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        freq = [0] * (k + 1)
        n = len(nums)
        for i in range(n // 2):
            a = nums[i]
            b = nums[n - i - 1]
            diff = abs(a - b)
            if diff <= k:
                freq[diff] += 1
        return (n // 2) - max(freq) if freq else n // 2