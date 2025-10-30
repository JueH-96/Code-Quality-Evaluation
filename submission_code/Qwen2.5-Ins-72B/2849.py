from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        
        for start in range(n):
            seen = set()
            current_imbalance = 0
            for end in range(start, n):
                num = nums[end]
                if num not in seen:
                    seen.add(num)
                    if num - 1 in seen:
                        current_imbalance -= 1
                    if num + 1 in seen:
                        current_imbalance -= 1
                    current_imbalance += 1
                total_imbalance += current_imbalance - 1
        
        return total_imbalance