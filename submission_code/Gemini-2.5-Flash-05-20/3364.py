import math
from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)

        # dp[i][j] = minimum sum to divide nums[i...n-1] into (m-j) subarrays,
        #              such that the bitwise AND conditions are met for andValues[j...m-1].
        # Base case: dp[n][m] = 0 (empty suffix nums[n...n-1], 0 remaining andValues)
        # All other dp[n][j] for j < m are float('inf') (cannot partition empty array into non-zero parts).
        # All other dp[i][m] for i < n are float('inf') (cannot use all andValues if nums still remains).
        
        # Initialize dp table with infinity.
        # dp dimensions: (n + 1) x (m + 1)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        
        # Base case
        dp[n][m] = 0

        # Iterate `j` from `m-1` down to `0`. `j` represents the current index in `andValues` we are trying to match.
        # We process `andValues` in reverse order because the state `dp[i][j]` depends on `dp[k+1][j+1]`.
        for j in range(m - 1, -1, -1):
            # Iterate `i` from `n-1` down to `0`. `i` represents the starting index in `nums` for the current segment.
            for i in range(n - 1, -1, -1):
                # `current_and` will store the bitwise AND of elements in the subarray `nums[i...k]`.
                # Initialize with all bits set, for numbers up to 10^5 (< 2^17), so 18 bits is safe.
                current_and = (1 << 18) - 1 
                
                # Iterate `k` from `i` to `n-1`. `k` is the ending index of the current subarray `nums[i...k]`.
                # This loop tries all possible end points for the current `j`-th subarray.
                for k in range(i, n):
                    # Update the running bitwise AND for the subarray nums[i...k].
                    current_and &= nums[k]
                    
                    # Check if the bitwise AND of the current subarray matches the required `andValues[j]`.
                    if current_and == andValues[j]:
                        # If a valid partition exists for the remaining part of `nums` (from `k+1` to `n-1`)
                        # using the remaining `andValues` (from `j+1` to `m-1`).
                        if dp[k+1][j+1] != float('inf'):
                            # Update `dp[i][j]` with the minimum sum.
                            # `nums[k]` is the value (last element) of the current subarray `nums[i...k]`.
                            dp[i][j] = min(dp[i][j], dp[k+1][j+1] + nums[k])
                    
                    # Optimization: If `current_and` becomes strictly less than `andValues[j]`,
                    # it's impossible to achieve `andValues[j]` by further ANDing operations,
                    # because AND operations can only decrease or maintain the value, never increase it.
                    # This effectively prunes the search space for `k`.
                    # This is key for the O(log(MAX_VAL)) factor for positive andValues[j].
                    # For andValues[j] = 0, current_and will never be < 0, so this check doesn't prune.
                    if current_and < andValues[j]:
                        break
        
        # The result is stored in dp[0][0], representing the minimum sum for partitioning
        # the entire `nums` array (`nums[0...n-1]`) using all `andValues` (`andValues[0...m-1]`).
        result = dp[0][0]
        
        # If result is still infinity, it means no valid partition was found.
        if result == float('inf'):
            return -1
        else:
            return result