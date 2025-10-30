class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        candidates = []
        for i in range(len(grid)):
            # Sort each row in descending order
            sorted_row = sorted(grid[i], reverse=True)
            # Take elements up to the limit for this row
            candidates.extend(sorted_row[:limits[i]])
        # Sort all candidates in descending order
        candidates.sort(reverse=True)
        # Take the first k elements and sum them
        return sum(candidates[:k])