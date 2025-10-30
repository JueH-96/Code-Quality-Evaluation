from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        def backtrack(index: int, count: int) -> bool:
            if count == len(nums):
                return True
            if index == len(nums):
                return False
            for i in range(index, len(nums)):
                total = sum(nums[index:i+1])
                if total >= m or i == len(nums) - 1:
                    if backtrack(i + 1, count + 1):
                        return True
            return False

        return backtrack(0, 0)