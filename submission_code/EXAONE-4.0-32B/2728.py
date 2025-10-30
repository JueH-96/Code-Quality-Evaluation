class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort(reverse=True)
        
        max_col = max(len(row) for row in nums)
        score = 0
        
        for j in range(max_col):
            max_val = 0
            for row in nums:
                if j < len(row):
                    if row[j] > max_val:
                        max_val = row[j]
            score += max_val
        
        return score