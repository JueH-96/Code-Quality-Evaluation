class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums:
            max_nums = [max(row) for row in nums]
            score += max(max_nums)
            for i in range(len(nums)):
                nums[i].remove(max_nums[i])
            nums = [row for row in nums if row]
        return score