import heapq
import math
from collections import defaultdict
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        if k * x > n:
            return -1 # Should not happen based on constraints

        costs = []

        # Sliding window median and cost calculation
        # Heaps: `left_heap_actual` (max-heap, stores smaller half, negative),
        # `right_heap_actual` (min-heap, stores larger half)
        # To calculate cost = sum |a_i - median|, median is the standard median.
        # We need left size floor(x/2), right size ceil(x/2). Median is right[0].
        # Or left size ceil(x/2), right size floor(x/2). Median is -left[0].
        # Let's use left size ceil(x/2), right size floor(x/2). Median is -left[0].
        left_heap_actual = [] # Max-heap
        right_heap_actual = [] # Min-heap
        left_sum_actual = 0
        right_sum_actual = 0
        removed_counts_actual = defaultdict(int)

        target_left_size = math.ceil(x / 2)
        target_right_size = math.floor(x / 2)

        def prune_actual():
            while left_heap_actual and -left_heap_actual[0] in removed_counts_actual and removed_counts_actual[-left_heap_actual[0]] > 0:
                val = -left_heap_actual[0]
                removed_counts_actual[val] -= 1
                if removed_counts_actual[val] == 0:
                    del removed_counts_actual[val]
                heapq.heappop(left_heap_actual)
                left_sum_actual -= val

            while right_heap_actual and right_heap_actual[0] in removed_counts_actual and removed_counts_actual[right_heap_actual[0]] > 0:
                val = right_heap_actual[0]
                removed_counts_actual[val] -= 1
                if removed_counts_actual[val] == 0:
                    del removed_counts_actual[val]
                heapq.heappop(right_heap_actual)
                right_sum_actual -= val

        def rebalance_sizes():
            prune_actual()
            while len(left_heap_actual) > target_left_size:
                m = -heapq.heappop(left_heap_actual)
                left_sum_actual -= m
                heapq.heappush(right_heap_actual, m)
                right_sum_actual += m
                prune_actual()

            prune_actual()
            while len(right_heap_actual) > target_right_size:
                m = heapq.heappop(right_heap_actual)
                right_sum_actual -= m
                heapq.heappush(left_heap_actual, -m)
                left_sum_actual += m
                prune_actual()

            prune_actual()
            while len(left_heap_actual) < target_left_size and right_heap_actual:
                m = heapq.heappop(right_heap_actual)
                right_sum_actual -= m
                heapq.heappush(left_heap_actual, -m)
                left_sum_actual += m
                prune_actual()

            prune_actual()
            while len(right_heap_actual) < target_right_size and left_heap_actual:
                m = -heapq.heappop(left_heap_actual)
                left_sum_actual -= m
                heapq.heappush(right_heap_actual, m)
                left_sum_actual += m
                prune_actual()


        def add_num_median(num):
            prune_actual()
            # Add to heap maintaining median property approximately
            # If num <= current median (-left[0]), add to left. Else add to right.
            # Handle empty heap case: add first element to left.
            # If left empty or num <= -left_heap_actual[0]: # This requires left_heap_actual[0] to be accessible.
            # A simpler rule for adding while maintaining balance:
            # Add to the heap that has fewer elements, to maintain size balance.
            if len(left_heap_actual) <= len(right_heap_actual):
                 heapq.heappush(left_heap_actual, -num)
                 left_sum_actual += num
            else:
                 heapq.heappush(right_heap_actual, num)
                 right_sum_actual += num
            
            # After adding, ensure max(left) <= min(right)
            prune_actual()
            if left_heap_actual and right_heap_actual and -left_heap_actual[0] > right_heap_actual[0]:
                 l_val = -heapq.heappop(left_heap_actual)
                 left_sum_actual -= l_val
                 r_val = heapq.heappop(right_heap_actual)
                 right_sum_actual -= r_val
                 heapq.heappush(left_heap_actual, -r_val)
                 left_sum_actual += r_val
                 heapq.heappush(right_heap_actual, l_val)
                 right_sum_actual += l_val
            
            rebalance_sizes()


        def remove_num_median(num):
             removed_counts_actual[num] += 1


        # Initialize window
        for i in range(x):
             add_num_median(nums[i]) # Use the add logic based on this setup

        # Calculate cost for the first window
        prune_actual()
        # Median for sum |a_i - median| is the standard median.
        # With left size ceil(x/2) and right size floor(x/2), median is -left_heap_actual[0].
        median = -left_heap_actual[0]
        cost = (len(left_heap_actual) - len(right_heap_actual)) * median + right_sum_actual - left_sum_actual
        costs.append(cost)


        # Slide window
        for i in range(1, n - x + 1):
            # Remove nums[i-1]
            remove_num_median(nums[i-1]) # Mark as removed

            # Add nums[i+x-1]
            add_num_median(nums[i+x-1]) # Add and rebalance

            # Calculate cost
            prune_actual() # Prune before accessing heap tops/sums
            median = -left_heap_actual[0]
            cost = (len(left_heap_actual) - len(right_heap_actual)) * median + right_sum_actual - left_sum_actual
            costs.append(cost)


        # --- End of Sliding Window ---

        # DP to find minimum cost for k non-overlapping subarrays
        m = len(costs) # Number of possible starting positions (n - x + 1)
        
        # dp[i] will store min cost for j blocks considering start indices 0..i
        # Use 1D array, dp[i] for j, prev_dp[i] for j-1

        # prev_dp stores dp[j-1][i] for i in 0..m-1
        prev_dp = [float('inf')] * m

        # Base case j=0: min cost for 0 blocks considering 0..i is 0 for i >= -1.
        # When computing for j=1, prev_dp[p] corresponds to dp[0][p].
        # dp[0][i] = 0 for i in 0..m-1.
        prev_dp = [0] * m # This represents dp[0][i] for i = 0..m-1.

        # Iterate for j from 1 to k
        for j in range(1, k + 1):
            curr_dp = [float('inf')] * m
            
            # Iterate for i from 0 to m-1 (possible start indices)
            for i in range(m):
                # Option 1: Don't use start index i for the j-th block
                # min cost considering starts 0..i is same as min cost considering starts 0..i-1.
                if i > 0:
                    curr_dp[i] = curr_dp[i-1]
                # else: curr_dp[0] remains inf initially for j > 0

                # Option 2: Use start index i for the j-th block
                # Need min cost for j-1 blocks considering start indices 0..i-x.
                prev_i = i - x
                
                cost_prev_j_minus_1 = float('inf')
                
                # We need dp[j-1][prev_i].
                # If j=1, need dp[0][prev_i]. dp[0][p] = 0 for p >= -1.
                if j == 1:
                     # If prev_i is conceptually >= -1 (meaning considering an empty set of indices or more), cost is 0.
                     # Indices are 0..m-1. prev_i refers to index in this range.
                     # prev_i = -1 means considering start indices up to -1 (empty set).
                     # So if prev_i >= -1, cost_prev_j_minus_1 = 0.
                     # Check against index in costs array (0..m-1)
                     if prev_i >= -1:
                          cost_prev_j_minus_1 = 0 # Cost for 0 blocks is 0

                else: # j > 1. Need dp[j-1][prev_i] from prev_dp.
                     # prev_dp stores dp[j-1][p] for p in 0..m-1.
                     if prev_i >= 0: # prev_i must be a valid index in prev_dp array (0..m-1)
                         cost_prev_j_minus_1 = prev_dp[prev_i]

                if cost_prev_j_minus_1 != float('inf'):
                     # Check if adding costs[i] makes sense. costs[i] is for index i. Always valid.
                     curr_dp[i] = min(curr_dp[i], cost_prev_j_minus_1 + costs[i])
            
            prev_dp = curr_dp # Update dp for the next iteration j+1

        # The result is the minimum cost to get k blocks considering start indices 0 to m-1
        final_answer = prev_dp[m-1]

        # If final_answer is still infinity, it means it's impossible to find k blocks.
        # Constraints k*x <= n ensures it's always possible.

        return final_answer