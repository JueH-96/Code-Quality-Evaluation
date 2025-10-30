class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums:
            max_nums = [max(row) for row in nums]
            max_num = max(max_nums)
            score += max_num
            for i in range(len(nums)):
                nums[i].remove(max_num)
                if not nums[i]:
                    nums.remove(nums[i])
        return score