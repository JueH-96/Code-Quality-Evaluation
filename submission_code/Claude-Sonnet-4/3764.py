class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        candidates = []
        
        # For each row, collect the top 'limits[i]' elements
        for i in range(len(grid)):
            row = grid[i]
            # Sort row in descending order and take at most limits[i] elements
            sorted_row = sorted(row, reverse=True)
            candidates.extend(sorted_row[:limits[i]])
        
        # Sort all candidates in descending order
        candidates.sort(reverse=True)
        
        # Take the top k elements
        return sum(candidates[:k])