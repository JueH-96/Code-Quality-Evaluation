class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        for i in range(len(nums)):
            nums[i].sort()
        
        for j in range(len(nums[0])):
            max_val = 0
            for i in range(len(nums)):
                max_val = max(max_val, nums[i][j])
            score += max_val
        
        return score