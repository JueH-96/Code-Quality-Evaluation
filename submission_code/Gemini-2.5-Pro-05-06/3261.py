import math

class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Initialize with a value representing negative infinity
        self.tree = [-float('inf')] * (2 * size)

    def update(self, index, value):
        # Update value at original index `index`
        pos = index + self.size
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = max(self.tree[pos * 2], self.tree[pos * 2 + 1])

    def query(self, left, right): # Query on interval [left, right)
        if left >= right: # Empty range
            return -float('inf')
        
        res = -float('inf')
        left += self.size
        right += self.size
        while left < right:
            if left % 2 == 1: # If left child is odd, it's a right child. Include it.
                res = max(res, self.tree[left])
                left += 1
            if right % 2 == 1: # If right child is odd, it's a right child. Exclude its parent later.
                right -= 1
                res = max(res, self.tree[right])
            left //= 2
            right //= 2
        return res

class Solution:
    def minOrAfterOperations(self, nums: list[int], k: int) -> int:
        n = len(nums)
        
        # Memoization for check function results (optional, likely not very effective here)
        # memo_check = {}

        def check(target_or_val: int) -> bool:
            # if target_or_val in memo_check:
            #    return memo_check[target_or_val]

            # dp[i] = max number of valid segments for prefix nums[0...i-1]
            # dp array indices: 0 to n. dp[0] for empty prefix. dp[i] for nums[0...i-1].
            # Initialize dp values to -infinity, dp[0] = 0.
            # (Segment tree handles initialization implicitly by tree nodes default to -inf)
            
            st = SegmentTree(n + 1) # Segment tree for dp values, indices 0 to n.
            st.update(0, 0) # dp[0] = 0

            # active_j_ands stores pairs (val, start_idx_k).
            # val = AND of nums[start_idx_k ... current_num_idx-1].
            # start_idx_k is the actual index in nums array for the start of this AND sequence.
            # This list is for the current prefix ending at nums[i-1].
            active_j_ands = [] 
            
            current_dp_i_val = -float('inf') # To store dp[i]

            for i in range(1, n + 1): # Iterate to compute dp[i] (for prefix nums[0...i-1])
                                      # current element processed is nums[i-1]
                current_num = nums[i-1]
                new_active_j_ands = []

                # Segment type 1: current_num forms a segment by itself: [nums[i-1]]
                # This segment starts at nums index i-1.
                # The dp state this relies on is dp[i-1].
                new_active_j_ands.append((current_num, i - 1)) 

                # Segment type 2: current_num extends segments ending at nums[i-2]
                # For each (v, k_prev) in active_j_ands (from previous iteration i-1):
                #   v = AND(nums[k_prev ... i-2])
                #   New segment AND value: v & current_num = AND(nums[k_prev ... i-1])
                #   Starting index remains k_prev.
                for val_prev, start_idx_k_prev in active_j_ands:
                    new_val = val_prev & current_num
                    if new_val == new_active_j_ands[-1][0]: # Compact if same AND value
                        new_active_j_ands[-1] = (new_val, start_idx_k_prev) # Keep earliest start_idx
                    else:
                        new_active_j_ands.append((new_val, start_idx_k_prev))
                active_j_ands = new_active_j_ands
                
                # Calculate dp[i] using the updated active_j_ands
                current_dp_i_val = -float('inf')
                
                # `last_k_for_dp_lookup` is the exclusive upper bound for dp query range.
                # A segment `nums[s_idx ... i-1]` corresponds to `dp[s_idx] + 1`.
                # `active_j_ands` is sorted by `start_idx_k` descending.
                # Example: active_j_ands = [(valA, kA), (valB, kB), ...] where kA > kB > ...
                # `valA` is AND sum for segments starting in `[kA, i-1]`. We query dp in `[kA, i-1]`.
                # `valB` is AND sum for segments starting in `[kB, kA-1]`. We query dp in `[kB, kA-1]`.
                last_dp_query_upper_bound = i 
                
                for val_seg, start_idx_k_seg in active_j_ands:
                    # `val_seg` is the AND sum for segments starting at `s_idx` in
                    # `[start_idx_k_seg, last_dp_query_upper_bound - 1]` and ending at `i-1`.
                    # We need `max(dp[s_idx])` for `s_idx` in this range.
                    if (val_seg | target_or_val) == target_or_val: # If segment value is valid
                        # Query for max(dp[k]) for k in [start_idx_k_seg, last_dp_query_upper_bound - 1].
                        # Segment tree query is [left, right).
                        max_prev_dp = st.query(start_idx_k_seg, last_dp_query_upper_bound)
                        if max_prev_dp > -float('inf'):
                            current_dp_i_val = max(current_dp_i_val, max_prev_dp + 1)
                    
                    last_dp_query_upper_bound = start_idx_k_seg # Next iteration uses this as exclusive upper bound

                st.update(i, current_dp_i_val)

            max_segments = current_dp_i_val # This is dp[n]
            
            if max_segments == -float('inf'): # Impossible to make all segments valid
                result = False
            else:
                num_ops_needed = n - max_segments
                result = num_ops_needed <= k
            
            # memo_check[target_or_val] = result
            return result

        # Binary search for the smallest target_or_val
        ans = (1 << 30) - 1 
        low = 0
        high = (1 << 30) - 1 

        while low <= high:
            mid = low + (high - low) // 2
            if mid < 0: # Should not happen with non-negative low/high
                mid = 0
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans