import heapq
import collections

class MedianAndSumTracker:
    def __init__(self):
        self.max_heap = [] # Stores (-val) for lower half elements (to simulate max-heap)
        self.min_heap = [] # Stores val for upper half elements
        self.lazy_removals = collections.Counter() # Tracks elements logically removed but still in heaps
        
        self.sum_max_heap = 0 # Sum of active (non-removed) elements in max_heap
        self.sum_min_heap = 0 # Sum of active (non-removed) elements in min_heap
        
        self.size_max_heap = 0 # Count of active elements in max_heap
        self.size_min_heap = 0 # Count of active elements in min_heap

    def _prune_heaps(self):
        # Remove elements from heap tops if they are marked for lazy removal.
        # This ensures heap[0] always refers to a logically active element.
        while self.max_heap and self.lazy_removals[-self.max_heap[0]] > 0:
            self.lazy_removals[-self.max_heap[0]] -= 1
            heapq.heappop(self.max_heap)
        while self.min_heap and self.lazy_removals[self.min_heap[0]] > 0:
            self.lazy_removals[self.min_heap[0]] -= 1
            heapq.heappop(self.min_heap)

    def _balance_heaps(self):
        self._prune_heaps() # Clean top elements before balancing
        
        # Ensure max_heap has at most one more element than min_heap
        # (This is for the median property when total elements 'x' is odd, median is in max_heap)
        while self.size_max_heap > self.size_min_heap + 1:
            val = -heapq.heappop(self.max_heap) # Move from max_heap to min_heap
            heapq.heappush(self.min_heap, val)
            
            self.sum_max_heap -= val
            self.size_max_heap -= 1
            self.sum_min_heap += val
            self.size_min_heap += 1
            self._prune_heaps() # A move might expose a lazy-removed element at new top

        # Ensure min_heap does not have more elements than max_heap
        while self.size_min_heap > self.size_max_heap:
            val = heapq.heappop(self.min_heap) # Move from min_heap to max_heap
            heapq.heappush(self.max_heap, -val)
            
            self.sum_min_heap -= val
            self.size_min_heap -= 1
            self.sum_max_heap += val
            self.size_max_heap += 1
            self._prune_heaps() # A move might expose a lazy-removed element at new top

    def add_num(self, val):
        # Determine which heap to add to based on the current median.
        # _prune_heaps() ensures that self.max_heap[0] is the true median candidate.
        self._prune_heaps() 
        if not self.max_heap or val <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -val)
            self.sum_max_heap += val
            self.size_max_heap += 1
        else:
            heapq.heappush(self.min_heap, val)
            self.sum_min_heap += val
            self.size_min_heap += 1
        self._balance_heaps()

    def remove_num(self, val):
        # Mark value for lazy removal
        self.lazy_removals[val] += 1
        
        # Decrement sum and size from the appropriate heap's counts/sums.
        # This decision is based on where the element logically belongs relative to the median.
        # This uses the current (pruned) median as a proxy, which is amortized correct.
        self._prune_heaps() 
        if val <= -self.max_heap[0]:
            self.sum_max_heap -= val
            self.size_max_heap -= 1
        else:
            self.sum_min_heap -= val
            self.size_min_heap -= 1
        self._balance_heaps()

    def get_cost(self):
        self._prune_heaps() # Ensure heaps are clean before calculating cost
        
        # If no elements, cost is infinite (should not happen for valid window)
        if not self.max_heap:
            return float('inf') 

        median_val = -self.max_heap[0]
        
        # Cost is sum(|num - median_val|) for elements in the window
        # This can be calculated as:
        # sum(median_val - num) for num <= median_val 
        #   + sum(num - median_val) for num > median_val
        # Which simplifies to:
        # (size_max_heap * median_val - sum_max_heap) + (sum_min_heap - size_min_heap * median_val)
        return (self.size_max_heap * median_val - self.sum_max_heap) + \
               (self.sum_min_heap - self.size_min_heap * median_val)


class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        N = len(nums)

        # Step 1: Precompute costs for all possible subarrays of size x
        # costs[i] will store the minimum operations for subarray nums[i : i+x]
        # This is done using a sliding window with the MedianAndSumTracker.
        costs = [0] * (N - x + 1)
        median_tracker = MedianAndSumTracker()

        # Initialize the tracker with the first window (nums[0 : x])
        for i in range(x):
            median_tracker.add_num(nums[i])
        costs[0] = median_tracker.get_cost()

        # Slide the window across the array
        for i in range(1, N - x + 1):
            median_tracker.remove_num(nums[i-1]) # Remove element leaving the window
            median_tracker.add_num(nums[i+x-1]) # Add element entering the window
            costs[i] = median_tracker.get_cost() # Get cost for the new window

        # Step 2: Dynamic Programming
        # dp[c][j] = minimum operations to have 'c' non-overlapping subarrays
        # using elements from nums[0] to nums[j-1].
        # The dp table size is (k+1) rows and (N+1) columns.
        dp = [[float('inf')] * (N + 1) for _ in range(k + 1)]

        # Base case: 0 subarrays cost 0 operations, regardless of the prefix length.
        for j in range(N + 1):
            dp[0][j] = 0

        # Fill the DP table
        for c in range(1, k + 1): # Iterate for number of subarrays from 1 to k
            for j in range(1, N + 1): # Iterate for prefix length from 1 to N
                # Option 1: Don't form the c-th subarray ending at nums[j-1].
                # In this case, the minimum cost is inherited from the previous prefix length.
                dp[c][j] = dp[c][j-1]

                # Option 2: Form the c-th subarray ending at nums[j-1].
                # This subarray is nums[j-x : j].
                # This option is only possible if there are at least 'x' elements in the current prefix (j >= x).
                if j >= x: 
                    # The cost for this current subarray (nums[j-x : j]) is precomputed.
                    current_subarray_cost = costs[j-x]
                    
                    # The previous (c-1) subarrays must end at or before nums[j-x-1].
                    # This corresponds to the state dp[c-1][j-x].
                    prev_subarrays_cost = dp[c-1][j-x]

                    # If the cost for the previous subarrays is not infinity, update dp[c][j].
                    if prev_subarrays_cost != float('inf'):
                        dp[c][j] = min(dp[c][j], prev_subarrays_cost + current_subarray_cost)
        
        # The final answer is the minimum cost for 'k' subarrays using all 'N' elements.
        return dp[k][N]