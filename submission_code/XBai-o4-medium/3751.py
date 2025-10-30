from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        original = nums.count(k)
        max_diff = 0
        for v in range(1, 51):
            current_max = -float('inf')
            current_sum = 0
            for num in nums:
                if num == v:
                    val = 1
                elif num == k:
                    val = -1
                else:
                    val = 0
                current_sum = max(val, current_sum + val)
                current_max = max(current_max, current_sum)
            if current_max > max_diff:
                max_diff = current_max
        return original + max(max_diff, 0)