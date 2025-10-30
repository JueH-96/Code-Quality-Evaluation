class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums[0]:
            max_nums = []
            for row in nums:
                max_num = max(row)
                max_nums.append(max_num)
                row.remove(max_num)
            score += max(max_nums)
        return score