from typing import List
from math import inf

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        if n == 0:
            return 0
        max_k = k
        if max_k == 0:
            return 0
        dp = [-float('inf')] * (max_k + 1)
        dp[0] = 0
        
        for row_index in range(n):
            current_row = grid[row_index]
            current_limit = limits[row_index]
            # Sort the current row in descending order
            sorted_row = sorted(current_row, reverse=True)
            # Generate prefix sums for this row, up to current_limit
            prefix = [0]
            current_sum = 0
            count = 0
            for num in sorted_row:
                if count >= current_limit:
                    break
                current_sum += num
                prefix.append(current_sum)
                count += 1
            
            # Update the dp array in reverse order
            for j in range(max_k, -1, -1):
                max_x = min(current_limit, j)
                if max_x == 0:
                    continue  # No possible x to consider
                best = dp[j]  # Default is taking 0 elements from this row
                # Iterate over possible x values to find the best one
                for x in range(1, max_x + 1):
                    prev_index = j - x
                    if prev_index < 0:
                        break
                    current_candidate = dp[prev_index] + prefix[x]
                    if current_candidate > best:
                        best = current_candidate
                if best > dp[j]:
                    dp[j] = best
        
        # Find the maximum value in dp[0..k]
        result = max(dp[:k+1])
        return result if result != -float('inf') else 0