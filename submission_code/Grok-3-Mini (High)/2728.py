class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        # Find the maximum length of the rows
        max_len = max(len(row) for row in nums)
        
        # Initialize score
        score = 0
        
        # For each index i from 0 to max_len-1
        for i in range(max_len):
            # Collect values at index i for rows that have it
            vals = [row[i] for row in nums if i < len(row)]
            if vals:  # Ensure there are values before finding max
                score += max(vals)
        
        return score