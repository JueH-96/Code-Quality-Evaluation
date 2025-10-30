import math
from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = 10**10
        dp = [[INF for _ in range(m + 1)] for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Precompute cumulative AND for each end index
        cum_and = [None] * n
        for e in range(n):
            f_list = [0] * (e + 1)
            f_list[e] = nums[e]
            for s in range(e - 1, -1, -1):
                f_list[s] = nums[s] & f_list[s + 1]
            cum_and[e] = f_list
        
        # DP to find minimum sum
        for k in range(1, m + 1):  # number of subarrays
            for pos in range(1, n + 1):  # number of elements covered
                end_idx = pos - 1
                targ = andValues[k - 1]
                f_list = cum_and[end_idx]
                
                # Binary search for the leftmost s where AND(s to end_idx) == targ
                low = 0
                high = end_idx
                left_idx = -1
                while low <= high:
                    mid = (low + high) // 2
                    if f_list[mid] == targ:
                        left_idx = mid
                        high = mid - 1  # Search left to find the first occurrence
                    elif f_list[mid] < targ:
                        low = mid + 1
                    else:
                        high = mid - 1
                
                if left_idx == -1:
                    continue  # No s satisfies the AND condition, skip this pos for current k
                
                # Binary search for the first s where f_list[s] > targ to find the right boundary
                low_ub = 0
                high_ub = end_idx
                result_ub = end_idx + 1  # Beyond the end, indicating no greater value
                while low_ub <= high_ub:
                    mid_ub = (low_ub + high_ub) // 2
                    if f_list[mid_ub] > targ:
                        result_ub = mid_ub
                        high_ub = mid_ub - 1  # Search left
                    else:
                        low_ub = mid_ub + 1
                
                R_idx = result_ub - 1  # The rightmost index where AND == targ
                
                # Now find the minimum of dp[s][k-1] for s in [left_idx, R_idx]
                min_prev = min(dp[s][k - 1] for s in range(left_idx, R_idx + 1))
                dp[pos][k] = min(dp[pos][k], min_prev + nums[end_idx])
        
        # Check the result
        if dp[n][m] < INF:
            return dp[n][m]
        else:
            return -1