class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        sorted_rows = [sorted(row, reverse=True) for row in nums]
        
        # Find the maximum number of operations needed
        max_length = max(len(row) for row in sorted_rows)
        
        score = 0
        for i in range(max_length):
            current_max = -1
            for row in sorted_rows:
                if i < len(row):
                    current_max = max(current_max, row[i])
            score += current_max
        return score