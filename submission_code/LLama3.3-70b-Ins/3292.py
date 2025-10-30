from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        marked = [False] * n
        for i in range(len(changeIndices)):
            if nums[changeIndices[i] - 1] > 0:
                nums[changeIndices[i] - 1] -= 1
            else:
                marked[changeIndices[i] - 1] = True
            if all(marked):
                return i + 1
        return -1 if not all(x == 0 for x in nums) else len(changeIndices)