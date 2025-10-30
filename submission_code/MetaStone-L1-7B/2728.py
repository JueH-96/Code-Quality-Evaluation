class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while any(row for row in nums if row):
            collected = []
            for row in nums:
                if not row:
                    continue
                max_val = max(row)
                collected.append(max_val)
            step_max = max(collected)
            score += step_max
            for i in range(len(nums)):
                row = nums[i]
                if not row:
                    continue
                idx = row.index(max_val)
                del row[idx]
        return score