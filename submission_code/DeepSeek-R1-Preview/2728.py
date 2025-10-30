class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        sorted_rows = []
        max_length = 0
        for row in nums:
            sorted_row = sorted(row, reverse=True)
            sorted_rows.append(sorted_row)
            if len(sorted_row) > max_length:
                max_length = len(sorted_row)
        
        score = 0
        # For each column index, find the maximum value across all rows
        for j in range(max_length):
            current_max = 0
            for row in sorted_rows:
                if j < len(row) and row[j] > current_max:
                    current_max = row[j]
            score += current_max
        
        return score