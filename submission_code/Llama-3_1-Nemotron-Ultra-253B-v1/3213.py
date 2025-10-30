from bisect import bisect_right
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        binary = [1 if num == max_val else 0 for num in nums]
        n = len(binary)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + binary[i]
        ans = 0
        for j in range(n):
            current_sum = prefix[j + 1]
            target = current_sum - k
            if target < 0:
                continue
            pos = bisect_right(prefix, target, 0, j + 1) - 1
            if pos >= 0:
                ans += (pos + 1)
        return ans