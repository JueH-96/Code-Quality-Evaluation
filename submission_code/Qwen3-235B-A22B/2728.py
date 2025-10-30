class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        score = 0
        max_steps = max(len(row) for row in nums)
        
        # Iterate through each 'column' (i.e., each step's elements)
        for j in range(max_steps):
            current_max = 0
            for row in nums:
                if j < len(row):
                    current_max = max(current_max, row[j])
            score += current_max
        
        return score