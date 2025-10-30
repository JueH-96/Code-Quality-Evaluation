import heapq
from collections import Counter
import math
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)

        class MedianCostSlidingWindow:
            def __init__(self, size_x):
                self.size_x = size_x
                self.max_heap = [] # Stores negative values of the smaller half (including median)
                self.min_heap = [] # Stores positive values of the larger half
                self.sum_max_heap = 0
                self.sum_min_heap = 0
                self.deleted_max = Counter()
                self.deleted_min = Counter()

                # Target sizes for heaps
                # max_heap should have size floor((x-1)/2) + 1
                # min_heap should have size x - (floor((x-1)/2) + 1)
                self.target_max_size = math.floor((self.size_x - 1) / 2) + 1
                self.target_min_size = self.size_x - self.target_max_size

            def _prune(self, heap, deleted_counter):
                while heap and abs(heap[0]) in deleted_counter and deleted_counter[abs(heap[0])] > 0:
                    deleted_counter[abs(heap[0])] -= 1
                    heapq.heappop(heap)

            def add(self, val):
                # Prune before peeking top
                self._prune(self.max_heap, self.deleted_max)
                
                # Decide which heap it should conceptually go into based on current state
                # If max_heap is empty, or val is smaller than or equal to the largest element in max_heap (negated top)
                if not self.max_heap or val <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -val)
                    self.sum_max_heap += val
                else:
                    heapq.heappush(self.min_heap, val)
                    self.sum_min_heap += val
                
                # Balance after adding
                self._balance()

            def remove(self, val):
                 # Prune before peeking top to make decisions based on valid tops
                 self._prune(self.max_heap, self.deleted_max)
                 self._prune(self.min_heap, self.deleted_min)

                 # Decide which heap it was likely in based on current state
                 # If val <= the largest element in max_heap (negated top)
                 # OR if min_heap is empty (implies val belongs to the max_heap side)
                 if not self.min_heap or (self.max_heap and val <= -self.max_heap[0]):
                     self.deleted_max[val] += 1
                     self.sum_max_heap -= val
                 else:
                     self.deleted_min[val] += 1
                     self.sum_min_heap -= val
                 # Balance is called after add/remove in the main loop, but remove needs sum/count updated first.
                 # Let's call balance explicitly here to simplify the main loop.
                 self._balance()


            def _balance(self):
                # Ensure heap lengths match target sizes after pruning deleted elements.

                while len(self.max_heap) < self.target_max_size:
                    self._prune(self.min_heap, self.deleted_min)
                    if not self.min_heap: break # Should not happen if window size > 0 and elements exist
                    val = heapq.heappop(self.min_heap)
                    self.sum_min_heap -= val
                    heapq.heappush(self.max_heap, -val)
                    self.sum_max_heap += val

                while len(self.min_heap) < self.target_min_size:
                    self._prune(self.max_heap, self.deleted_max)
                    if not self.max_heap: break # Should not happen if window size > 0 and elements exist
                    val = -heapq.heappop(self.max_heap)
                    self.sum_max_heap -= val
                    heapq.heappush(self.min_heap, val)
                    self.sum_min_heap += val

                # Prune again after potential moves
                self._prune(self.max_heap, self.deleted_max)
                self._prune(self.min_heap, self.deleted_min)


            def get_cost(self):
                # Ensure heaps are pruned to get correct top element
                self._prune(self.max_heap, self.deleted_max)
                self._prune(self.min_heap, self.deleted_min)

                # Median is the largest element in the smaller half (top of max_heap, negated)
                # After balancing, max_heap contains the smallest target_max_size elements, including the median.
                if not self.max_heap: 
                    # This case should not be reached if window size is x >= 2.
                    # However, for completeness, return a large value if window is empty.
                    return float('inf') 

                median = -self.max_heap[0]

                # Cost = sum |s_i - median| = sum(median - s_i) for s_i <= median + sum(s_i - median) for s_i > median
                # Elements <= median are the target_max_size elements conceptually in max_heap. Sum is sum_max_heap.
                # Elements > median are the target_min_size elements conceptually in min_heap. Sum is sum_min_heap.
                # Cost = (target_max_size * median - sum_max_heap) + (sum_min_heap - target_min_size * median)
                # Cost = sum_min_heap - sum_max_heap + (target_max_size - target_min_size) * median

                # Simplified formula derived earlier:
                # Cost = S_min - S_max + (2 * target_max_size - x) * median
                size_diff_term = (2 * self.target_max_size - self.size_x)
                cost = self.sum_min_heap - self.sum_max_heap + size_diff_term * median
                return cost

        window_calc = MedianCostSlidingWindow(x)
        costs = []

        # Calculate cost for the first window [0...x-1]
        for i in range(x):
            window_calc.add(nums[i])
        costs.append(window_calc.get_cost())

        # Calculate costs for subsequent windows [i...i+x-1]
        for i in range(1, n - x + 1):
            # Element nums[i-1] leaves, nums[i+x-1] enters
            window_calc.remove(nums[i-1]) # remove calls balance internally
            window_calc.add(nums[i+x-1]) # add calls balance internally
            costs.append(window_calc.get_cost())

        # Dynamic Programming
        # dp[i][j]: min cost using first i elements (nums[0...i-1]) to get exactly j non-overlapping subarrays of size x
        # Indices: i from 0 to n, j from 0 to k
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]

        # Base case: 0 cost for 0 subarrays
        for i in range(n + 1):
            dp[i][0] = 0

        # Fill DP table
        # i iterates through the prefix length (from 1 to n)
        # j iterates through the number of subarrays (from 1 to k)
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # Option 1: Don't form the j-th subarray ending at index i-1
                # The minimum cost is the same as using the first i-1 elements to get j subarrays.
                dp[i][j] = dp[i-1][j]

                # Option 2: Form the j-th subarray ending at index i-1
                # This subarray has size x and is nums[i-x ... i-1].
                # This is only possible if the prefix length i is at least x.
                if i >= x:
                    # The subarray starts at index i-x. Its cost is pre-calculated in `costs`.
                    window_start_index = i - x 
                    current_window_cost = costs[window_start_index]
                    
                    # The previous j-1 non-overlapping subarrays must be formed using the elements before this one,
                    # i.e., nums[0 ... i-x-1]. The length of this prefix is i-x.
                    # If it was possible to get j-1 subarrays in the prefix of length i-x:
                    if dp[i-x][j-1] != float('inf'):
                         # Update dp[i][j] with the minimum cost from this option
                         dp[i][j] = min(dp[i][j], dp[i-x][j-1] + current_window_cost)

        # The minimum cost to get exactly k non-overlapping subarrays using the whole array (prefix length n)
        # Constraints k*x <= n guarantee that it is always possible to place k non-overlapping subarrays.
        # The cost is always finite.
        return dp[n][k]