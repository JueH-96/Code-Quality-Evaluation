class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        m = len(nums)
        n = len(nums[0])

        for row in nums:
            row.sort()
        
        for j in range(n):
            mx = 0
            for i in range(m):
                mx = max(mx, nums[i][j])
            score += mx
        return score