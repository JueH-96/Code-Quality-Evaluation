class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        max_length = max(len(row) for row in nums)
        
        for i in range(max_length):
            current_max = 0
            for row in nums:
                if i < len(row):
                    current_max = max(current_max, row[i])
            score += current_max
        
        return score