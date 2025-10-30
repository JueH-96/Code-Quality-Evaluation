class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        candidates = []
        
        for i in range(len(grid)):
            row = grid[i]
            row_sorted = sorted(row, reverse=True)
            # Take at most limits[i] elements from this row
            for j in range(min(limits[i], len(row))):
                candidates.append(row_sorted[j])
        
        # Sort candidates in descending order and take the first k
        candidates.sort(reverse=True)
        
        return sum(candidates[:k])