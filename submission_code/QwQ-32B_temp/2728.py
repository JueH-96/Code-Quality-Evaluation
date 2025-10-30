from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        sorted_rows = []
        for row in nums:
            sorted_rows.append(sorted(row, reverse=True))
        
        # Determine the maximum length of rows to know how many steps
        max_len = max(len(row) for row in sorted_rows)
        
        score = 0
        
        # For each step (column index)
        for i in range(max_len):
            current_max = 0
            for row in sorted_rows:
                if i < len(row):
                    current_max = max(current_max, row[i])
            score += current_max
        
        return score