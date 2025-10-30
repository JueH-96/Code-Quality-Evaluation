from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        sorted_nums = [sorted(row, reverse=True) for row in nums]
        max_len = max(len(row) for row in sorted_nums) if sorted_nums else 0
        score = 0
        
        for i in range(max_len):
            current_max = 0
            for row in sorted_nums:
                if i < len(row) and row[i] > current_max:
                    current_max = row[i]
            score += current_max
        
        return score