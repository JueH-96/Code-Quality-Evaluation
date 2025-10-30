import bisect
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for i in range(n):
            sorted_list = []
            imbalance = 0
            for j in range(i, n):
                num = nums[j]
                pos = bisect.bisect_left(sorted_list, num)
                left = sorted_list[pos-1] if pos > 0 else None
                right = sorted_list[pos] if pos < len(sorted_list) else None
                prev_gap = 0
                if left is not None and right is not None:
                    if right - left > 1:
                        prev_gap = 1
                bisect.insort(sorted_list, num)
                new_gap = 0
                if left is not None:
                    if num - left > 1:
                        new_gap += 1
                if right is not None:
                    if right - num > 1:
                        new_gap += 1
                imbalance = imbalance - prev_gap + new_gap
                total += imbalance
        return total