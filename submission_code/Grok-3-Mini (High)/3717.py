import math
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (size + 1)
            
            def update(self, index, delta):
                while index <= self.size:
                    self.tree[index] += delta
                    index += index & -index
            
            def prefix_sum(self, index):
                res = 0
                idx = index
                while idx > 0:
                    res += self.tree[idx]
                    idx -= idx & -idx
                return res
        
        n = len(nums)
        m = n - x + 1  # Number of possible subarrays of size x
        OFFSET = 1000000
        MAX_FT_INDEX = 2000002  # To handle values from -10^6 to 10^6 with offset and 1-based indexing
        INF = 2000000000000  # A large number to represent infinity, larger than max possible cost
        
        # Compute cost list for each subarray of size x
        cost_list = [0] * m
        ft_freq = FenwickTree(MAX_FT_INDEX)
        ft_sum = FenwickTree(MAX_FT_INDEX)
        window_sum = 0
        
        # Add first x elements to Fenwick trees and window sum
        for i in range(x):
            val = nums[i]
            idx_ft = val + OFFSET + 1
            ft_freq.update(idx_ft, 1)
            ft_sum.update(idx_ft, val)
            window_sum += val
        
        # Slide the window and compute cost for each starting position
        for start in range(m):
            # Find the median value m_val
            k_th = (x // 2) + 1  # 1-based index for k-th smallest (choosing a median that minimizes sum of abs deviations)
            left, right = -1000000, 1000000
            while left <= right:
                mid = (left + right) // 2
                num_leq_mid = ft_freq.prefix_sum(mid + OFFSET + 1)
                if num_leq_mid >= k_th:
                    right = mid - 1
                else:
                    left = mid + 1
            m_val = left  # Smallest V such that num <= V >= k_th
            
            # Compute necessary sums and counts for cost calculation
            if m_val - 1 >= -1000000:
                P_less = (m_val - 1) + OFFSET + 1  # Index for prefix sum up to val <= m_val - 1
                num_less_m = ft_freq.prefix_sum(P_less)
                sum_less_m = ft_sum.prefix_sum(P_less)
            else:
                num_less_m = 0
                sum_less_m = 0
            P_leq_m = m_val + OFFSET + 1  # Index for prefix sum up to val <= m_val
            num_leq_m = ft_freq.prefix_sum(P_leq_m)
            sum_leq_m = ft_sum.prefix_sum(P_leq_m)
            num_greater_m = x - num_leq_m
            sum_greater_m = window_sum - sum_leq_m
            
            # Calculate cost for the current subarray
            cost = m_val * num_less_m - sum_less_m + sum_greater_m - m_val * num_greater_m
            cost_list[start] = cost
            
            # Slide the window to the next position if not the last
            if start < m - 1:
                val_remove = nums[start]
                idx_ft_remove = val_remove + OFFSET + 1
                ft_freq.update(idx_ft_remove, -1)
                ft_sum.update(idx_ft_remove, -val_remove)
                window_sum -= val_remove
                
                val_add = nums[start + x]
                idx_ft_add = val_add + OFFSET + 1
                ft_freq.update(idx_ft_add, 1)
                ft_sum.update(idx_ft_add, val_add)
                window_sum += val_add
        
        # Dynamic programming to find min cost to place exactly k subarrays
        dp = [[INF for _ in range(m)] for _ in range(k + 1)]  # dp[j][i]: min cost to place j subarrays with j-th starting at i
        
        # Base case: j=1, cost to place one subarray starting at i
        for i in range(m):
            dp[1][i] = cost_list[i]
        
        # Fill DP table for j from 2 to k
        for j in range(2, k + 1):
            prev_dp = dp[j - 1]
            # Compute prefix minimum of prev_dp
            prefix_min_list = [INF] * m
            run_min = INF
            for i in range(m):
                run_min = min(run_min, prev_dp[i])
                prefix_min_list[i] = run_min  # prefix_min_list[i] = min of prev_dp[0] to prev_dp[i]
            
            # Compute dp[j][i] for each i
            for i in range(m):
                if i < (j - 1) * x:
                    dp[j][i] = INF  # Cannot place j-th subarray starting at i if too early
                else:
                    dp[j][i] = cost_list[i] + prefix_min_list[i - x]  # Min over previous starts <= i - x
        
        # The answer is the minimum over all possible ending positions for k subarrays
        return min(dp[k])