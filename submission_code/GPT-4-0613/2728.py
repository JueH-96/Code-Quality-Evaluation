class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums:
            max_nums = []
            for row in nums:
                max_num = max(row)
                row.remove(max_num)
                max_nums.append(max_num)
            score += max(max_nums)
            nums = [row for row in nums if row]
        return score