class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort()
        nums = list(map(list, zip(*nums)))
        score = sum(max(row) for row in nums)
        return score