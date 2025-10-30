import heapq
import collections
import math
from typing import List

# Define INF for calculations. Using float('inf') is typical in Python.
INF = float('inf')

# Class implementing sliding window median calculation using two heaps with lazy deletion.
# It efficiently calculates the minimum cost (sum of absolute differences from the median)
# for a sliding window of size x.
class SlidingWindowMedian:
    def __init__(self, x):
        self.x = x  # Window size
        # lo: max heap (stores negative numbers to simulate max heap), holds smaller half
        # hi: min heap, holds larger half
        self.lo, self.hi = [], []  
        # Sums of elements in each logical partition
        self.sum_lo, self.sum_hi = 0, 0
        # Dictionary to track counts of elements marked for deletion (lazy deletion)
        self.deleted = collections.defaultdict(int)
        # Actual number of non-deleted elements in each partition
        self.lo_size, self.hi_size = 0, 0
        # Target number of elements for each partition based on window size x
        # lo partition gets ceil(x/2), hi partition gets floor(x/2) elements.
        self.lo_target_size = (x + 1) // 2
        self.hi_target_size = x // 2

    # Removes elements marked for deletion from the top of the specified heap ('lo' or 'hi').
    def _prune(self, heap_type):
        if heap_type == 'lo':
            heap = self.lo
            # Iterate while heap is not empty and top element is marked for deletion
            while heap and self.deleted[-heap[0]] > 0:
                val = -heapq.heappop(heap) # Pop and get original value
                self.deleted[val] -= 1 # Decrement deletion count
                if self.deleted[val] == 0: del self.deleted[val] # Clean up dictionary if count is 0
        else: # heap_type == 'hi'
            heap = self.hi
            while heap and self.deleted[heap[0]] > 0:
                val = heapq.heappop(heap)
                self.deleted[val] -= 1
                if self.deleted[val] == 0: del self.deleted[val]

    # Rebalances the heaps to maintain target sizes and partition property.
    # Ensures lo contains the smallest ceil(x/2) elements and hi the largest floor(x/2).
    def _balance(self):
        # Loop until heaps are balanced according to target sizes.
        while True:
             # Prune heaps first to work with actual elements
            self._prune('lo')
            self._prune('hi')

            # Condition 1: lo partition has too many elements. Move max element from lo to hi.
            if self.lo_size > self.lo_target_size:
                if not self.lo: break # Safety check
                move_val = -heapq.heappop(self.lo)
                self.sum_lo -= move_val; self.lo_size -= 1
                heapq.heappush(self.hi, move_val)
                self.sum_hi += move_val; self.hi_size += 1
            # Condition 2: hi partition has too many elements. Move min element from hi to lo.
            elif self.hi_size > self.hi_target_size:
                if not self.hi: break # Safety check
                move_val = heapq.heappop(self.hi)
                self.sum_hi -= move_val; self.hi_size -= 1
                heapq.heappush(self.lo, -move_val)
                self.sum_lo += move_val; self.lo_size += 1
            # If neither condition met, heaps should be balanced assuming total size x maintained.
            else:
                break

    # Adds a value to the structure.
    def add(self, val):
        # Prune 'lo' before checking its top for placement decision
        self._prune('lo') 
        
        # Place 'val' into appropriate heap based on current median (-lo[0])
        # If lo is empty or val <= current median, put in lo. Otherwise, hi.
        if not self.lo or val <= -self.lo[0]:
             heapq.heappush(self.lo, -val)
             self.sum_lo += val; self.lo_size += 1
        else:
             heapq.heappush(self.hi, val)
             self.sum_hi += val; self.hi_size += 1
        self._balance() # Rebalance after addition

    # Marks a value for removal (lazy deletion).
    def remove(self, val):
        # Increment deletion count for this value
        self.deleted[val] += 1 
        
        # Adjust logical sizes and sums based on where 'val' *would* reside
        # This step assumes 'val' was indeed present.
        self._prune('lo') # Prune first to get potentially updated median
        
        # Determine partition based on current median. Check for empty lo.
        median = -self.lo[0] if self.lo else INF 
        
        if val <= median: # Belongs to lo partition
            self.sum_lo -= val; self.lo_size -= 1
        else: # Belongs to hi partition
            self.sum_hi -= val; self.hi_size -= 1

        self._balance() # Rebalance after logical removal

    # Returns the minimum cost (sum of absolute differences from median) for the current window.
    def get_median_cost(self):
        # Ensure heaps are balanced and tops valid before calculation
        self._balance() 
        
        # If lo is empty, implies window size is 0 or structure error. Return 0 cost for empty window.
        if not self.lo: return 0 
        
        # Median is the largest element in the lo partition (top of max heap)
        median = -self.lo[0]
        
        # Calculate cost using the formula: Sum(|v - median|) = Sum_{v in lo} (median - v) + Sum_{v in hi} (v - median)
        # This simplifies to (lo_size * median - sum_lo) + (sum_hi - hi_size * median)
        cost = (self.lo_size * median - self.sum_lo) + (self.sum_hi - self.hi_size * median)
        return cost


class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        # N is the maximum starting index for a subarray of size x. Indices range from 0..N.
        N = n - x 
        
        # If n < x, no valid subarrays exist. Problem constraints guarantee n >= x.
        if N < 0: return -1 # Should not be reached given constraints

        # Calculate costs for all possible subarrays using the sliding window median approach
        costs = [0] * (N + 1)
        swm = SlidingWindowMedian(x)
        
        # Handle edge case x=0 if constraints allowed it. Constraints say x>=2.
        # if x == 0: ...

        # Initialize window with first x elements
        # Ensure x > 0 before proceeding, guaranteed by constraints.
        for i in range(x):
            swm.add(nums[i])
        costs[0] = swm.get_median_cost() # Cost for subarray nums[0:x]

        # Slide window from index 1 to N
        for i in range(1, N + 1):
            swm.remove(nums[i-1]) # Remove element leaving window
            swm.add(nums[i+x-1]) # Add element entering window
            costs[i] = swm.get_median_cost() # Cost for subarray nums[i:i+x]

        # Dynamic programming setup
        # dp[i][j]: min cost to select j subarrays, with the j-th subarray *starting* at index i.
        # PMin[i][j]: min cost to select j subarrays, with the j-th subarray starting at any index p <= i.
        # Initialize DP tables with infinity
        dp = [[INF] * (k + 1) for _ in range(N + 1)]
        PMin = [[INF] * (k + 1) for _ in range(N + 1)]

        # Base case: j = 1 (selecting 1 subarray)
        for i in range(N + 1):
            dp[i][1] = costs[i]
            # Calculate prefix minimum for j=1
            PMin[i][1] = min(PMin[i-1][1], dp[i][1]) if i > 0 else dp[i][1]

        # Fill DP table for j = 2..k
        for j in range(2, k + 1):
            for i in range(N + 1):
                # A subarray starting at i requires the previous (j-1)th subarray
                # to start at an index p <= i-x.
                # Check if index i allows for a previous non-overlapping subarray (i.e., i >= x)
                if i >= x:
                    # Minimum cost for j-1 subarrays ending at or before index i-x
                    prev_min_cost = PMin[i-x][j-1] 
                    # If it was possible to form j-1 subarrays (cost is not INF)
                    if prev_min_cost != INF: 
                        # Total cost is cost of current subarray + min cost of previous j-1 subarrays
                        dp[i][j] = costs[i] + prev_min_cost 
                
                # Update prefix minimum PMin[i][j]
                # Minimum is either the previous prefix minimum or the current dp value
                PMin[i][j] = min(PMin[i-1][j], dp[i][j]) if i > 0 else dp[i][j]
        
        # Final answer is the minimum cost to achieve k subarrays ending at any valid position up to N
        final_min_cost = PMin[N][k]
        
        # Constraints guarantee k*x <= n, so a solution always exists. Check for INF just in case.
        # The problem asks for integer result. Our costs are sums/diffs of integers, so result is integer.
        return int(final_min_cost) if final_min_cost != INF else -1