class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        marked = [False] * (n + 1)
        marked[0] = True
        for i in range(m):
            nums[changeIndices[i] - 1] -= 1
            if nums[changeIndices[i] - 1] == 0:
                marked[changeIndices[i]] = True
            if all(marked):
                return i + 1
        return -1