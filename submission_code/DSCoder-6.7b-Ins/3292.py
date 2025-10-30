class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        marked = [False] * n
        indices_set = set(changeIndices)
        for i in range(m):
            if nums[changeIndices[i]-1] == 0:
                marked[changeIndices[i]-1] = True
            for j in range(n):
                if nums[j] > 0:
                    nums[j] -= 1
            if all(marked):
                return i+1
        return -1