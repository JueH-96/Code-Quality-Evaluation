from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Calculate prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        min_sum = float('inf')
        # Iterate through all possible starting indices
        for i in range(n):
            # Determine the maximum ending index for the current start
            max_j = min(i + r - 1, n - 1)
            # Iterate through all possible ending indices starting from i+l-1 to max_j
            for j in range(i + l - 1, max_j + 1):
                # Calculate the sum of the subarray from i to j using prefix sums
                current_sum = prefix_sum[j+1] - prefix_sum[i]
                # Update min_sum if the current sum is positive and smaller than the current min_sum
                if current_sum > 0 and current_sum < min_sum:
                    min_sum = current_sum
        # If no valid subarray found, return -1
        return min_sum if min_sum != float('inf') else -1