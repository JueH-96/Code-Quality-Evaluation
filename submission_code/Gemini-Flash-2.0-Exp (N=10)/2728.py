class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while nums and any(row for row in nums):
            max_vals = []
            for row in nums:
                if row:
                    max_val = max(row)
                    max_vals.append(max_val)
                    row.remove(max_val)
            if max_vals:
                score += max(max_vals)
        return score