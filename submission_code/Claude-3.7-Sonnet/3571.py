from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        # Compute the longest increasing path ending at each point
        dp_before = [1] * n
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                if coordinates[j][0] < coordinates[i][0] and coordinates[j][1] < coordinates[i][1]:
                    dp_before[i] = max(dp_before[i], dp_before[j] + 1)
        
        # Sort indices in descending order of coordinates for dp_after computation
        # This ensures we compute dp_after[j] before dp_after[i] if j can come after i
        sorted_indices = sorted(range(n), key=lambda idx: (-coordinates[idx][0], -coordinates[idx][1]))
        
        # Compute the longest increasing path starting from each point
        dp_after = [1] * n
        for idx in sorted_indices:
            for j in range(n):
                if j == idx:
                    continue
                if coordinates[idx][0] < coordinates[j][0] and coordinates[idx][1] < coordinates[j][1]:
                    dp_after[idx] = max(dp_after[idx], dp_after[j] + 1)
        
        # The maximum path length including coordinates[k] is the sum of
        # the longest path ending at k and the longest path starting from k,
        # minus 1 to avoid counting k twice
        return dp_before[k] + dp_after[k] - 1