class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        max_cols = max(len(row) for row in nums)
        
        # For each column position (after sorting)
        for col in range(max_cols):
            max_in_operation = 0
            # Get the element from each row at this column position
            for row in nums:
                if col < len(row):
                    max_in_operation = max(max_in_operation, row[col])
            
            score += max_in_operation
        
        return score