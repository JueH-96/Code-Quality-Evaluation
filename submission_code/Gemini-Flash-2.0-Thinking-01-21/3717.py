from typing import List
import heapq
from collections import Counter
import math # For float('inf')

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)

        # --- Sliding Window Cost Calculation using two heaps ---
        # max_heap stores negated values of the smaller half
        # min_heap stores values of the larger half
        # We maintain |max_heap| = ceil(x/2), |min_heap| = floor(x/2)
        max_heap = [] # stores -val
        min_heap = [] # stores val
        sum_max_heap = 0 # sum of original values in max_heap
        sum_min_heap = 0 # sum of original values in min_heap
        removed = Counter() # Lazy removal counter for elements leaving the window

        # Helper function to balance the heaps and clean removed elements
        def balance_heaps():
            nonlocal sum_max_heap, sum_min_heap

            # Clean removed elements from the tops first
            while max_heap and removed[-max_heap[0]] > 0:
                removed[-max_heap[0]] -= 1
                sum_max_heap -= -heapq.heappop(max_heap)
            while min_heap and removed[min_heap[0]] > 0:
                removed[min_heap[0]] -= 1
                sum_min_heap -= heapq.heappop(min_heap)

            # Balance sizes
            # max_heap should have size ceil(x/2), min_heap should have size floor(x/2)
            while len(max_heap) > (x + 1) // 2:
                val = -heapq.heappop(max_heap)
                sum_max_heap -= val
                heapq.heappush(min_heap, val)
                sum_min_heap += val
            while len(min_heap) > x // 2:
                val = heapq.heappop(min_heap)
                sum_min_heap -= val
                heapq.heappush(max_heap, -val)
                sum_max_heap += val
            # The size constraints ensure |max_heap| + |min_heap| = x.
            # If one heap is too small, the other must be too large.
            # The loops above handle the case where one heap is too large.
            # We don't need explicit checks for too small because they imply the other is too large.
            # However, let's add them for clarity and correctness in edge cases,
            # although the previous loops should implicitly cover this when x >= 2.
            # The case where len(max_heap) < (x + 1) // 2 implies len(min_heap) > x // 2 + (x+1)//2 - len(max_heap) - (x//2) = x - len(max_heap).
            # If len(max_heap) = (x+1)//2 - 1, then len(min_heap) = x - ((x+1)//2 - 1) = x - (x+1)//2 + 1.
            # If x is odd, x=2m+1, (x+1)//2 = m+1, x//2=m. len(max_heap)=m, len(min_heap) = 2m+1 - m = m+1. Need len(max)=m+1.
            # If x is even, x=2m, (x+1)//2 = m, x//2=m. len(max_heap)=m-1, len(min_heap) = 2m - (m-1) = m+1. Need len(max)=m.
            # The loops below ensure the smaller heap gets elements if the larger one was already correct size.
            while len(max_heap) < (x + 1) // 2 and min_heap:
                 val = heapq.heappop(min_heap)
                 sum_min_heap -= val
                 heapq.heappush(max_heap, -val)
                 sum_max_heap += val
            while len(min_heap) < x // 2 and max_heap:
                 val = -heapq.heappop(max_heap)
                 sum_max_heap -= val
                 heapq.heappush(min_heap, val)
                 sum_min_heap += val


            # Clean tops again after balancing sizes
            while max_heap and removed[-max_heap[0]] > 0:
                removed[-max_heap[0]] -= 1
                sum_max_heap -= -heapq.heappop(max_heap)
            while min_heap and removed[min_heap[0]] > 0:
                removed[min_heap[0]] -= 1
                sum_min_heap -= heapq.heappop(min_heap)

            # Ensure median property: max(max_heap) <= min(min_heap)
            while max_heap and min_heap and -max_heap[0] > min_heap[0]:
                 val_max = -heapq.heappop(max_heap)
                 sum_max_heap -= val_max
                 val_min = heapq.heappop(min_heap)
                 sum_min_heap -= val_min

                 heapq.heappush(max_heap, -val_min)
                 sum_max_heap += val_min
                 heapq.heappush(min_heap, val_max)
                 sum_min_heap += val_max

                 # Clean tops again after swapping
                 while max_heap and removed[-max_heap[0]] > 0:
                    removed[-max_heap[0]] -= 1
                    sum_max_heap -= -heapq.heappop(max_heap)
                 while min_heap and removed[min_heap[0]] > 0:
                    removed[min_heap[0]] -= 1
                    sum_min_heap -= heapq.heappop(min_heap)


        # Helper function to add a value to the window
        def add(val):
            nonlocal sum_max_heap, sum_min_heap
            # Add to max_heap first, then balance
            heapq.heappush(max_heap, -val)
            sum_max_heap += val
            balance_heaps()

        # Helper function to mark a value for removal from the window
        def remove(val):
            removed[val] += 1 # Mark as removed
            balance_heaps() # Rebalance sizes and clean tops

        # Helper function to calculate the cost for the current window
        # This function should be called *after* add/remove and balance_heaps
        def get_cost():
            # Ensure heaps are clean and balanced before calculating cost
            # balance_heaps() # Called by add/remove already, not strictly needed here

            # The median is the largest element in the max_heap.
            # If x >= 2, max_heap should not be empty after balancing.
            # Check only needed for x=0 or x=1, but constraints say x>=2.
            # if not max_heap: return float('inf')

            m = -max_heap[0]

            # Cost = sum_max_heap + sum_min_heap + (len(max_heap) - len(min_heap)) * m
            return sum_max_heap + sum_min_heap + (len(max_heap) - len(min_heap)) * m

        # Precompute costs for all possible windows of size x
        window_costs = []
        # Initialize for the first window nums[0:x]
        for i in range(x):
            add(nums[i])
        window_costs.append(get_cost())

        # Slide the window
        # Iterate through the starting indices of the sliding window, from 1 to n-x.
        # The loop `for i in range(x, n)` iterates through the *ending* indices of the windows, from x-1 to n-1.
        # When the window ends at index `i`, it starts at index `i-x+1`.
        # This means the loop calculates costs for windows starting at `x-x+1=1`, ..., `n-1-x+1 = n-x`.
        # Combined with the initial cost for start index 0, we cover all start indices from 0 to n-x.
        for i in range(x, n):
            # Remove nums[i-x] (element leaving the window from the left)
            remove(nums[i-x])
            # Add nums[i] (element entering the window from the right)
            add(nums[i])
            # Calculate cost for the new window starting at index i-x+1
            # The index in window_costs corresponds to the starting index of the window.
            # The current window starts at index i-x+1.
            window_costs.append(get_cost())


        # --- Dynamic Programming Calculation ---
        # dp[i][j] = minimum cost to obtain exactly j non-overlapping subarrays of size x
        # using elements from nums[0] up to nums[i-1].
        # i ranges from 0 to n (length of prefix)
        # j ranges from 0 to k (number of subarrays)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]

        # Base case: 0 subarrays cost 0 operations
        for i in range(n + 1):
            dp[i][0] = 0

        # Iterate through prefix lengths (from 1 up to n)
        for i in range(1, n + 1):
            # Iterate through number of subarrays (from 1 up to k)
            for j in range(1, k + 1):
                # Option 1: Do not end the j-th subarray at index i-1.
                # The minimum cost to get j subarrays using the prefix of length i
                # is at least the minimum cost to get j subarrays using the prefix of length i-1.
                dp[i][j] = dp[i-1][j]

                # Option 2: End the j-th subarray at index i-1.
                # This subarray must be nums[i-x : i].
                # This is only possible if the prefix is long enough, i.e., i >= x.
                # The start index of this subarray is i - x.
                # The cost of this subarray is window_costs[i-x].
                # The previous j-1 subarrays must end strictly before index i-x.
                # This means they must use elements up to index (i-x) - 1.
                # The minimum cost for the previous j-1 subarrays ending by index i-x-1
                # is dp[i-x][j-1].
                if i >= x:
                    current_window_start_index = i - x # Window nums[i-x ... i-1]
                    cost_current_window = window_costs[current_window_start_index]

                    # Update dp[i][j] only if a valid state for j-1 subarrays exists
                    if dp[i-x][j-1] != float('inf'):
                         dp[i][j] = min(dp[i][j], dp[i-x][j-1] + cost_current_window)

        # The final answer is the minimum cost to get exactly k subarrays using the entire array (prefix of length n).
        # The problem asks for "at least k". Based on the DP structure and examples,
        # finding the minimum cost for *exactly* k non-overlapping ones is likely the intended solution.
        # If a solution with >k subarrays were cheaper, this DP wouldn't find it directly,
        # but the constraints (small k, k*x <= n) suggest that finding the optimal exactly-k solution is sufficient.
        # Let's return dp[n][k].
        return dp[n][k]