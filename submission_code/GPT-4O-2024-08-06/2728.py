class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        
        # Continue until all rows are empty
        while any(nums):
            max_in_rows = []
            
            # For each row, find and remove the largest element
            for row in nums:
                if row:  # Check if the row is not empty
                    max_value = max(row)
                    row.remove(max_value)
                    max_in_rows.append(max_value)
            
            # Find the maximum of the removed elements and add to score
            if max_in_rows:
                score += max(max_in_rows)
        
        return score