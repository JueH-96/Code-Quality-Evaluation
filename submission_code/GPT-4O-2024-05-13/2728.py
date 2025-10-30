class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        
        # Continue until the matrix is empty
        while nums:
            max_in_rows = []
            
            # For each row, find and remove the largest element
            for row in nums:
                max_in_rows.append(max(row))
                row.remove(max(row))
            
            # Add the maximum of these largest elements to the score
            score += max(max_in_rows)
            
            # Remove empty rows
            nums = [row for row in nums if row]
        
        return score