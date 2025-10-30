from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def canAchieve(target):
            count, current_and = 0, (1 << 30) - 1
            for num in nums:
                current_and &= num
                if current_and & target == target:
                    count += 1
                    current_and = (1 << 30) - 1
            return len(nums) - count <= k
        
        result = (1 << 30) - 1
        for bit in range(29, -1, -1):
            if canAchieve(result ^ (1 << bit)):
                result ^= (1 << bit)
        return result