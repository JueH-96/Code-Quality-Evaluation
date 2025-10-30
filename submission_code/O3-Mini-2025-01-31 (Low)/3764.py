from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        candidates = []
        # For each row, collect the top candidates respecting the limit for that row.
        for i, row in enumerate(grid):
            limit = limits[i]
            # If limit is 0, skip this row.
            if limit <= 0:
                continue
            # Sort the row in descending order and take up to 'limit' elements.
            sorted_row = sorted(row, reverse=True)
            candidates.extend(sorted_row[:limit])
        
        # Sort the collection of candidates in descending order.
        candidates.sort(reverse=True)
        
        # We need to select at most k elements.
        # If there are fewer than k candidates, we sum them all.
        return sum(candidates[:k])
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    grid1 = [[1,2],[3,4]]
    limits1 = [1,2]
    k1 = 2
    print(sol.maxSum(grid1, limits1, k1))  # Output: 7

    # Example 2
    grid2 = [[5,3,7],[8,2,6]]
    limits2 = [2,2]
    k2 = 3
    print(sol.maxSum(grid2, limits2, k2))  # Output: 21