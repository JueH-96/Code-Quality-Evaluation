class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        max_cols = max(len(row) for row in nums)
        
        for col in range(max_cols):
            removed = []
            
            # Get the largest remaining element from each row
            for row in nums:
                if col < len(row):
                    removed.append(row[col])
            
            # Add the maximum of removed elements to the score
            if removed:
                score += max(removed)
        
        return score