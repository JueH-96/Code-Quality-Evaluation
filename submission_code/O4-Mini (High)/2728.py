class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row so we can easily pop off the largest element
        for row in nums:
            row.sort()
        
        score = 0
        # Continue until every row is empty
        while any(len(row) > 0 for row in nums):
            max_removed = 0
            # From each non‐empty row remove its current largest and track the max
            for row in nums:
                if row:
                    val = row.pop()
                    if val > max_removed:
                        max_removed = val
            score += max_removed
        
        return score