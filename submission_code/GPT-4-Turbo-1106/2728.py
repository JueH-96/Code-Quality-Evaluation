class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums:
            max_values = [max(row) for row in nums if row]
            highest = max(max_values)
            score += highest
            nums = [[num for num in row if num != highest] for row in nums]
        return score