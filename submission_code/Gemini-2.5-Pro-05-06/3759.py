class BIT:
    def __init__(self, size: int):
        # size is the maximum value that can be stored (M_VAL)
        # tree array is 1-indexed conceptually, so size+1 length
        self.tree = [0] * (size + 1)
        self.size = size # Max value that can be an index in BIT

    def update(self, val_idx: int, delta: int):
        # val_idx is 1-based value from nums2
        while val_idx <= self.size:
            self.tree[val_idx] += delta
            val_idx += val_idx & (-val_idx) # Add LSB

    def query_prefix_sum(self, val_idx: int) -> int:
        # Sum of elements with value <= val_idx
        s = 0
        while val_idx > 0:
            s += self.tree[val_idx]
            val_idx -= val_idx & (-val_idx) # Subtract LSB
        return s

class Solution:
    def findMaxSum(self, nums1: list[int], nums2: list[int], k: int) -> list[int]:
        n = len(nums1)
        if n == 0:
            return []

        M_VAL = 0
        for x in nums2:
            if x > M_VAL:
                M_VAL = x
        # Constraints: nums2[i] >= 1, n >= 1. So M_VAL will be at least 1.

        P = []
        for i in range(n):
            P.append((nums1[i], nums2[i], i))
        
        # Sort by nums1 values. Default tuple sort uses next elements for ties.
        P.sort() 
        
        ans = [0] * n
        
        bit_count = BIT(M_VAL)
        bit_sum = BIT(M_VAL)
        
        current_P_idx = 0
        while current_P_idx < n:
            group_start_idx = current_P_idx
            # Find end of block with same nums1 value
            while current_P_idx < n - 1 and P[current_P_idx][0] == P[current_P_idx+1][0]:
                current_P_idx += 1
            group_end_idx = current_P_idx
            
            # BITs currently contain nums2[j] for nums1[j] < P[group_start_idx][0]
            # Calculate answer for all elements in this block P[group_start_idx ... group_end_idx]
            
            num_elements_in_bit = bit_count.query_prefix_sum(M_VAL)
            current_block_ans_sum = 0
            
            if num_elements_in_bit > 0 and k > 0: # k is positive integer (>=1 by constraints)
                actual_k = min(k, num_elements_in_bit)
                
                # Binary search for V_thresh_val: the smallest value among the top actual_k values.
                # This means actual_k values are >= V_thresh_val.
                # And (num_elements_in_bit - actual_k) values are < V_thresh_val.
                
                # Initialize V_thresh_val. If all elements are 1 and actual_k > 0, V_thresh_val is 1.
                # Can also be set to 0 and if loop doesn't run (e.g. high<low), it stays 0.
                # But values are >=1. So if actual_k > 0, V_thresh_val must be >=1.
                V_thresh_val = 1 
                
                # Binary search range for values [1, M_VAL]
                low = 1
                high = M_VAL
                
                while low <= high:
                    mid_val = low + (high - low) // 2
                    if mid_val == 0: # Should not happen if M_VAL >= 1
                        low = 1 # Ensure progress if mid_val somehow becomes 0
                        continue
                    
                    # Count elements with value >= mid_val
                    count_ge_mid = num_elements_in_bit - bit_count.query_prefix_sum(mid_val - 1)
                    
                    if count_ge_mid >= actual_k:
                        # mid_val could be V_thresh_val, or an even larger value could be.
                        # We want the largest such mid_val that satisfies this.
                        V_thresh_val = mid_val
                        low = mid_val + 1 
                    else:
                        # mid_val is too high (not enough elements >= mid_val). Need smaller mid_val.
                        high = mid_val - 1
                
                # Now V_thresh_val is the smallest value that is part of the top actual_k sum.
                # Sum of values strictly greater than V_thresh_val
                sum_gt_Vthresh = bit_sum.query_prefix_sum(M_VAL) - bit_sum.query_prefix_sum(V_thresh_val)
                # Count of values strictly greater than V_thresh_val
                count_gt_Vthresh = num_elements_in_bit - bit_count.query_prefix_sum(V_thresh_val)
                
                # We need (actual_k - count_gt_Vthresh) more elements. 
                # These must all be equal to V_thresh_val.
                needed_count_of_Vthresh_val = actual_k - count_gt_Vthresh
                current_block_ans_sum = sum_gt_Vthresh + needed_count_of_Vthresh_val * V_thresh_val
            
            # Assign the calculated sum to all original indices in this block
            for i_block in range(group_start_idx, group_end_idx + 1):
                original_idx = P[i_block][2]
                ans[original_idx] = current_block_ans_sum
            
            # Add nums2 values from this processed block into BITs for next iterations
            for i_block in range(group_start_idx, group_end_idx + 1):
                val2_to_add = P[i_block][1]
                bit_count.update(val2_to_add, 1)
                bit_sum.update(val2_to_add, val2_to_add)
            
            current_P_idx = group_end_idx + 1 # Move to the start of the next block
            
        return ans