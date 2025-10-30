from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList
        
        def imbalance_count(subarr):
            sarr = SortedList(subarr)
            return sum(1 for i in range(len(sarr) - 1) if sarr[i+1] - sarr[i] > 1)
        
        total_imbalance = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                total_imbalance += imbalance_count(nums[i:j+1])
        return total_imbalance