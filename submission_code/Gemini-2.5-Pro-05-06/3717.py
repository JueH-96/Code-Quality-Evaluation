import heapq
from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)

        # costs[i] = min operations for nums[i...i+x-1]
        costs = [0] * (n - x + 1) 
        
        # Heaps for sliding window median
        h_low = []  # Max-heap for lower half (stores -val)
        h_high = [] # Min-heap for upper half (stores val)
        
        current_sum_low = 0
        current_sum_high = 0
        s_low = 0  # Effective size of h_low
        s_high = 0 # Effective size of h_high

        # For lazy deletion
        removed_from_low = Counter()
        removed_from_high = Counter()

        # Helper to clean top of a heap from lazily deleted elements
        def clean_heap_top(heap, removed_map, is_max_heap):
            while heap:
                # Get actual value stored in heap top
                val_in_heap_raw = heap[0] 
                # Convert to original number sense (max-heap stores negated)
                val_to_check = -val_in_heap_raw if is_max_heap else val_in_heap_raw
                
                if removed_map[val_to_check] > 0:
                    removed_map[val_to_check] -= 1
                    heapq.heappop(heap)
                else:
                    break # Top element is valid
        
        # Helper to balance heaps: s_low should be s_high or s_high + 1
        def balance_heaps():
            nonlocal current_sum_low, current_sum_high, s_low, s_high
            
            # Case 1: h_low is too large
            while s_low > s_high + 1:
                clean_heap_top(h_low, removed_from_low, True)
                # This break should ideally not be hit if s_low implies elements exist
                if not h_low: break 
                val = -heapq.heappop(h_low) # Pop from h_low
                s_low -= 1
                current_sum_low -= val
                
                heapq.heappush(h_high, val) # Push to h_high
                s_high += 1
                current_sum_high += val
            
            # Case 2: h_high is too large (or s_low is too small)
            while s_high > s_low:
                clean_heap_top(h_high, removed_from_high, False)
                if not h_high: break
                val = heapq.heappop(h_high) # Pop from h_high
                s_high -= 1
                current_sum_high -= val

                heapq.heappush(h_low, -val) # Push to h_low
                s_low += 1
                current_sum_low += val
            
            # Ensure tops are clean for subsequent median queries
            clean_heap_top(h_low, removed_from_low, True)
            clean_heap_top(h_high, removed_from_high, False)

        # Initialize with the first window nums[0...x-1]
        for j in range(x):
            val_to_add = nums[j]
            clean_heap_top(h_low, removed_from_low, True) # Clean before checking median
            
            if not h_low or val_to_add <= -h_low[0]: # If h_low empty or val fits in lower part
                heapq.heappush(h_low, -val_to_add)
                s_low += 1
                current_sum_low += val_to_add
            else:
                heapq.heappush(h_high, val_to_add)
                s_high += 1
                current_sum_high += val_to_add
            balance_heaps() # Balance after each addition
            
        # Calculate cost for the first window (nums[0...x-1])
        clean_heap_top(h_low, removed_from_low, True) # Ensure median is accurate
        # Due to constraints (x >= 2) and balancing, s_low >= 1, so h_low is not empty.
        median = -h_low[0] 
        costs[0] = (s_low * median - current_sum_low) + (current_sum_high - s_high * median)

        # Slide window across nums
        for i in range(n - x): # Window nums[i...i+x-1] slides to nums[i+1...i+x]
            val_out = nums[i]    # Element leaving
            val_in = nums[i+x]   # Element entering

            # Process val_out (mark for removal)
            clean_heap_top(h_low, removed_from_low, True) # Clean for current median
            current_median_of_old_window = -h_low[0] 

            if val_out <= current_median_of_old_window:
                removed_from_low[val_out] += 1
                s_low -= 1
                current_sum_low -= val_out
            else:
                removed_from_high[val_out] += 1
                s_high -= 1
                current_sum_high -= val_out
            balance_heaps() # Rebalance after conceptual removal

            # Process val_in (add to heaps)
            clean_heap_top(h_low, removed_from_low, True) # Clean for median decision for val_in
            
            if not h_low or val_in <= -h_low[0]:
                heapq.heappush(h_low, -val_in)
                s_low += 1
                current_sum_low += val_in
            else:
                heapq.heappush(h_high, val_in)
                s_high += 1
                current_sum_high += val_in
            balance_heaps() # Rebalance after conceptual addition

            # Calculate cost for the new window nums[i+1...i+x]
            clean_heap_top(h_low, removed_from_low, True)
            median = -h_low[0]
            costs[i+1] = (s_low * median - current_sum_low) + (current_sum_high - s_high * median)
        
        # --- DP Part ---
        # dp[p][i] = min cost for p subarrays, p-th subarray starts at index i
        dp = [[float('inf')] * (n - x + 1) for _ in range(k + 1)]

        # Base case: p=1
        for i in range(n - x + 1):
            dp[1][i] = costs[i]

        # Fill DP table for p from 2 to k
        for p_count in range(2, k + 1):
            # min_val_prev_dp_prefix[j_idx] stores min(dp[p_count-1][j] for j <= j_idx)
            # where j must be a valid start for p_count-1 blocks.
            min_val_prev_dp_prefix = [float('inf')] * (n - x + 1)
            current_min_dp_val = float('inf')
            
            # Smallest starting index for (p_count-1) blocks
            min_j_for_prev_p = (p_count - 2) * x 
            
            for j_scan in range(min_j_for_prev_p, n - x + 1):
                current_min_dp_val = min(current_min_dp_val, dp[p_count-1][j_scan])
                min_val_prev_dp_prefix[j_scan] = current_min_dp_val

            # Smallest starting index for p_count blocks
            min_i_for_current_p = (p_count - 1) * x
            for i in range(min_i_for_current_p, n - x + 1):
                # The (p_count-1)-th block must start at j <= i-x.
                idx_for_prev_min = i - x
                # Check if idx_for_prev_min is a valid index and in range for min_val_prev_dp_prefix
                if idx_for_prev_min >= min_j_for_prev_p: 
                    val_from_prev_dp = min_val_prev_dp_prefix[idx_for_prev_min]
                    if val_from_prev_dp != float('inf'): # And costs[i] is not inf (it shouldn't be)
                        dp[p_count][i] = costs[i] + val_from_prev_dp
        
        final_min_ops = float('inf')
        # Smallest starting index for k blocks
        min_i_for_k_blocks = (k - 1) * x
        for i in range(min_i_for_k_blocks, n - x + 1):
            final_min_ops = min(final_min_ops, dp[k][i])
            
        return int(final_min_ops)