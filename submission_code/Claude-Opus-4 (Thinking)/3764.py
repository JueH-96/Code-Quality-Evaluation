class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        candidates = []
        
        # For each row, collect the top limits[i] elements
        for i in range(len(grid)):
            row = grid[i]
            # Sort the row in descending order
            sorted_row = sorted(row, reverse=True)
            # Take at most limits[i] elements from this row
            num_to_take = min(limits[i], len(sorted_row))
            candidates.extend(sorted_row[:num_to_take])
        
        # Sort all candidates in descending order
        candidates.sort(reverse=True)
        
        # Take the top k elements and return their sum
        return sum(candidates[:k])