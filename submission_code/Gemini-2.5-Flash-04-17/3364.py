import sys
from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)

        # dp[i][j] = minimum sum using nums[0...i-1] to form j subarrays.
        # i: number of elements used from nums (from 0 to n).
        # j: number of subarrays formed (from 0 to m).
        # dp table size (n + 1) x (m + 1)

        # Using a large number for infinity, larger than any possible sum.
        # Max possible sum is n * max(nums) approx 10^4 * 10^5 = 10^9.
        # sys.maxsize is sufficiently large. Use // 2 to avoid potential overflow during additions.
        INF = sys.maxsize // 2

        # Initialize dp table with infinity
        dp = [[INF] * (m + 1) for _ in range(n + 1)]

        # Base case: 0 elements used, 0 subarrays formed, sum is 0.
        dp[0][0] = 0

        # Iterate over the number of subarrays j
        for j in range(1, m + 1):
            # Iterate over the number of elements used from nums, ending at index i-1
            # To form j subarrays using nums[0...i-1], we need at least j elements.
            # So, i ranges from j to n.
            for i in range(j, n + 1):
                # Calculate dp[i][j]
                # Consider the j-th subarray ending at index i-1.
                # Let its start index be k_start. The subarray is nums[k_start ... i-1].
                # The previous j-1 subarrays must partition nums[0 ... k_start - 1].
                # This previous partition uses k_start elements.
                # The minimum sum for this previous partition is found in dp[k_start][j-1].

                # The number of elements used for the previous j-1 subarrays is k_start.
                # To form j-1 subarrays, we need at least j-1 elements. So k_start >= j-1.
                # The last subarray starts at k_start and ends at i-1.
                # k_start can be as large as i-1 (single element subarray nums[i-1...i-1]).
                # So, k_start ranges from j-1 up to i-1.

                current_and_for_segment = (1 << 20) - 1 # Initialize with all bits set for a reasonable range

                # Iterate k_start from i-1 down to j-1
                # range(start, stop, step) includes start, up to (but not including) stop.
                # To include j-1, stop should be (j-1)-1 = j-2.
                # The loop range is [i-1, i-2, ..., j-1].
                for k_start in range(i - 1, j - 2, -1):
                    # The current segment being considered as the j-th subarray is nums[k_start ... i-1]
                    current_and_for_segment &= nums[k_start]

                    # Optimization: If current_and is already less than the target andValue[j-1],
                    # it will only decrease or stay the same as k_start decreases further (adding elements to the left).
                    # It can never become equal to the target andValue[j-1].
                    if current_and_for_segment < andValues[j - 1]:
                        break

                    # Check if the bitwise AND of the segment nums[k_start ... i-1] equals andValues[j-1]
                    if current_and_for_segment == andValues[j - 1]:
                        # This segment nums[k_start ... i-1] is a valid j-th subarray
                        # ending at index i-1 and starting at index k_start.
                        # The previous j-1 subarrays partition nums[0 ... k_start - 1],
                        # using k_start elements.
                        # The minimum sum for this previous partition is dp[k_start][j-1].

                        # If dp[k_start][j-1] is reachable (not infinity)
                        if dp[k_start][j - 1] != INF:
                            # Update dp[i][j] with the minimum sum found so far for partitioning
                            # nums[0...i-1] into j subarrays.
                            # The sum for this specific partition is the sum for the previous
                            # j-1 subarrays (dp[k_start][j-1]) plus the value of the j-th subarray,
                            # which is its last element, nums[i-1].
                            dp[i][j] = min(dp[i][j], dp[k_start][j - 1] + nums[i - 1])

        # The final answer is the minimum sum using all n elements (nums[0...n-1]) for m subarrays.
        result = dp[n][m]

        # If the result is still infinity, it means it's not possible to partition
        # nums according to the given conditions.
        return result if result != INF else -1