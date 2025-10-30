class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        rows = len(nums)
        if rows == 0:
            return 0
        cols = 0
        for row in nums:
            cols = max(cols, len(row))

        for i in range(rows):
            nums[i].sort(reverse=True)

        for _ in range(cols):
            removed_nums = []
            for i in range(rows):
                if nums[i]:
                    removed_nums.append(nums[i].pop(0))
            if removed_nums:
                score += max(removed_nums)
        return score