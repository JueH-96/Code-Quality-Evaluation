from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # We'll collect for each row only the candidates that we are actually allowed to pick.
        # Since all grid elements are nonnegative, it is always optimal for each row to consider the top
        # "limits[i]" values (i.e. the best ones from that row). Once we have these candidates from each row,
        # the overall best selection is simply to pick the highest k values with respect to the candidate list.
        candidates = []
        for i, row in enumerate(grid):
            # Sort the row in descending order.
            sorted_row = sorted(row, reverse=True)
            # Only consider at most limits[i] elements.
            num_to_take = limits[i]
            # Since limits[i] <= len(row) by constraints, we can take the top num_to_take elements.
            candidates.extend(sorted_row[:num_to_take])
        
        # Now, since we can only pick k elements in total and every candidate is independent,
        # the maximum sum is obtained by selecting the k highest candidates.
        candidates.sort(reverse=True)
        return sum(candidates[:k])