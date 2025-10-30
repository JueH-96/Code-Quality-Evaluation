from bisect import bisect_left, insort
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        total_imbalance = 0
        n = len(nums)
        for start in range(n):
            sorted_list = []
            imbalance = 0
            for end in range(start, n):
                num = nums[end]
                pos = bisect_left(sorted_list, num)
                insort(sorted_list, num)
                
                pred = sorted_list[pos-1] if pos > 0 else None
                succ = sorted_list[pos+1] if pos+1 < len(sorted_list) else None
                
                if pred is not None and num - pred > 1:
                    imbalance += 1
                if succ is not None and succ - num > 1:
                    imbalance += 1
                if pred is not None and succ is not None and succ - pred > 1:
                    imbalance -= 1
                total_imbalance += imbalance
        return total_imbalance