class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums and all(nums[i] for i in range(len(nums))):
            max_values = []
            for row in nums:
                if row:
                    max_val = max(row)
                    max_values.append(max_val)
                    row.remove(max_val)
            score += max(max_values)
        return score