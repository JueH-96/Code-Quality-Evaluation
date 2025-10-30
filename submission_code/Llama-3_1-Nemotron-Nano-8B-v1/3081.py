from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        stack = []
        count = 0
        for num in nums:
            if stack and stack[-1] < num:
                stack.pop()
                count += 1
            else:
                stack.append(num)
        return len(nums) - 2 * count