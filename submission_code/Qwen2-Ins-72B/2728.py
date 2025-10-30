class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        # Transpose the matrix to make it easier to find the max in each column
        transposed = list(zip(*nums))
        
        # For each column, find the max and add it to the score
        for col in transposed:
            score += max(col)
        
        return score