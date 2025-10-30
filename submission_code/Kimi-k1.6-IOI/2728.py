class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        max_cols = max(len(row) for row in nums)
        score = 0
        
        # Iterate over each column index
        for col in range(max_cols):
            current_max = 0
            # Check each row for the current column
            for row in nums:
                if col < len(row):
                    current_max = max(current_max, row[col])
            score += current_max
        
        return score