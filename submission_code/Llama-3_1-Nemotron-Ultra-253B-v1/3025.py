from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if total < target:
            return -1
        
        required = [0] * 31
        for i in range(31):
            required[i] = (target >> i) & 1
        
        available = [0] * 31
        for num in nums:
            bit = num.bit_length() - 1
            available[bit] += 1
        
        ops = 0
        
        for i in range(30, -1, -1):
            if required[i] == 0:
                continue
            
            if available[i] >= required[i]:
                available[i] -= required[i]
            else:
                deficit = required[i] - available[i]
                needed_splits = (deficit + 1) // 2
                if i + 1 >= 31 or available[i + 1] < needed_splits:
                    return -1
                ops += needed_splits
                available[i + 1] -= needed_splits
                available[i] += 2 * needed_splits
                available[i] -= deficit
        
        return ops