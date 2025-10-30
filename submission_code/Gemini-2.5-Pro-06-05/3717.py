import heapq
from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        """
        Calculates the minimum operations to have k non-overlapping subarrays of size x with equal elements.

        The solution is broken down into two main parts:
        1.  Pre-calculation of costs for all possible subarrays of size x.
        2.  Dynamic programming to find the minimum total cost for k non-overlapping subarrays.

        Part 1: Cost Calculation (Sliding Window Median)
        - The minimum operations to make a subarray's elements equal is to change them all to the subarray's median.
        - A sliding window approach with two heaps is used to efficiently calculate the median and cost for each window.
          - `lowers`: A max-heap for the smaller half of the window.
          - `highers`: A min-heap for the larger half.
        - Lazy deletion is used to handle element removal from the heaps.
        - This pre-computation takes O(n * log(x)) time.

        Part 2: Dynamic Programming
        - Let dp[j][i] be the minimum cost for j subarrays, with the j-th one starting at index i.
        - The recurrence is: dp[j][i] = costs[i] + min(dp[j-1][p]) for 0 <= p <= i-x.
        - This can be optimized by maintaining a running minimum of the previous DP row.
        - The DP has a time complexity of O(k * n) and can be space-optimized to O(n).
        
        The overall complexity is O(n * log(x) + k * n).
        """
        n = len(nums)
        m = n - x + 1

        if k == 0:
            return 0

        # 1. Precompute costs for all possible subarrays of size x
        costs = self._calculate_all_costs(nums, x)

        # 2. DP to find min cost for k subarrays
        # prev_dp[i] = min cost for j subarrays, with the j-th starting at index i
        # curr_dp[i] = min cost for j+1 subarrays, with the (j+1)-th starting at index i
        prev_dp = list(costs)

        for j in range(2, k + 1):
            curr_dp = [float('inf')] * m
            min_val = float('inf')
            
            start_i = (j - 1) * x
            for i in range(start_i, m):
                # Update running minimum from the previous DP row
                # The previous subarray must end before the current one starts, so it must start at or before i-x.
                min_val = min(min_val, prev_dp[i - x])
                
                if min_val != float('inf'):
                    curr_dp[i] = costs[i] + min_val
            
            prev_dp = curr_dp

        result = min(prev_dp)
        return int(result) if result != float('inf') else 0

    def _calculate_all_costs(self, nums: List[int], x: int) -> List[int]:
        n = len(nums)
        m = n - x + 1
        costs = [0] * m

        lowers, highers = [], []  # max-heap, min-heap
        l_sum, h_sum = 0, 0
        l_size, h_size = 0, 0
        to_remove = Counter()

        # Initialize with the first window
        initial_window = sorted(nums[:x])
        median_split_idx = (x + 1) // 2
        
        for val in initial_window[:median_split_idx]:
            heapq.heappush(lowers, -val)
            l_sum += val
        l_size = len(lowers)
        
        for val in initial_window[median_split_idx:]:
            heapq.heappush(highers, val)
            h_sum += val
        h_size = len(highers)
        
        median = -lowers[0]
        costs[0] = (median * l_size - l_sum) + (h_sum - median * h_size)

        # Slide the window
        for i in range(1, m):
            out_val = nums[i - 1]
            in_val = nums[i + x - 1]
            
            # 1. Remove out_val (conceptually) by marking it for lazy deletion
            to_remove[out_val] += 1
            if out_val <= -lowers[0]:
                l_size -= 1
                l_sum -= out_val
            else:
                h_size -= 1
                h_sum -= out_val
            
            # 2. Add in_val
            # The median might have shifted. Add relative to the current partition boundary.
            if lowers and in_val > -lowers[0]:
                heapq.heappush(highers, in_val)
                h_sum += in_val
                h_size += 1
            else:
                heapq.heappush(lowers, -in_val)
                l_sum += in_val
                l_size += 1

            # 3. Balance heaps by size, accounting for lazy deletions
            while l_size > h_size + 1:
                val = -heapq.heappop(lowers)
                if to_remove[val] > 0:
                    to_remove[val] -= 1; continue
                heapq.heappush(highers, val)
                l_sum -= val; h_sum += val
                l_size -= 1; h_size += 1

            while h_size > l_size:
                val = heapq.heappop(highers)
                if to_remove[val] > 0:
                    to_remove[val] -= 1; continue
                heapq.heappush(lowers, -val)
                h_sum -= val; l_sum += val
                h_size -= 1; l_size += 1

            # 4. Prune lazy-deleted elements from the tops of the heaps
            while lowers and to_remove[-lowers[0]] > 0:
                to_remove[-heapq.heappop(lowers)] -= 1
            while highers and to_remove[highers[0]] > 0:
                to_remove[heapq.heappop(highers)] -= 1

            # 5. Calculate cost for the new window
            median = -lowers[0]
            costs[i] = (median * l_size - l_sum) + (h_sum - median * h_size)
            
        return costs