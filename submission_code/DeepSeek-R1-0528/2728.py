class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        max_cols = max(len(row) for row in nums)
        
        for j in range(max_cols):
            current_max = 0
            for row in nums:
                if j < len(row):
                    current_max = max(current_max, row[j])
            score += current_max
        
        return score