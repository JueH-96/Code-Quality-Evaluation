from bisect import bisect_right
from typing import List

class Solution:
  def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
    max_val_in_nums = 0
    # Determine the maximum value in nums to set array sizes.
    # Constraints: nums.length >= 2, nums[i] >= 1. So max_val_in_nums >= 1.
    for x in nums:
        if x > max_val_in_nums:
            max_val_in_nums = x

    # 1. Frequency Count
    freq = [0] * (max_val_in_nums + 1)
    for x in nums:
        freq[x] += 1

    # 2. Count Multiples
    num_multiples = [0] * (max_val_in_nums + 1)
    for k in range(1, max_val_in_nums + 1):
        for multiple_val in range(k, max_val_in_nums + 1, k):
            # freq[multiple_val] is 0 if multiple_val was not in nums.
            num_multiples[k] += freq[multiple_val]
            
    # 3. Count GCDs
    count_gcd_is_g = [0] * (max_val_in_nums + 1)
    # Iterate g downwards from max_val_in_nums to 1
    for g in range(max_val_in_nums, 0, -1):
        if num_multiples[g] < 2:
            # Cannot form a pair if less than 2 multiples of g exist in nums
            count_gcd_is_g[g] = 0
            continue
        
        # Total pairs (x,y) where x and y are multiples of g
        pairs_where_both_are_multiples_of_g = num_multiples[g] * (num_multiples[g] - 1) // 2
        
        # Subtract pairs whose GCD is k*g where k > 1 (e.g., 2g, 3g, ...).
        # These counts (count_gcd_is_g[2g], etc.) have already been computed.
        sum_of_counts_for_larger_gcds = 0
        for multiple_of_g_val in range(2 * g, max_val_in_nums + 1, g):
            sum_of_counts_for_larger_gcds += count_gcd_is_g[multiple_of_g_val]
            
        count_gcd_is_g[g] = pairs_where_both_are_multiples_of_g - sum_of_counts_for_larger_gcds

    # 4. Prefix Sums and Queries
    # ps_arr[idx] stores sum of counts for GCDs 1...(idx+1)
    # So ps_arr[g_val-1] = sum_{k=1 to g_val} count_gcd_is_g[k]
    ps_arr = [] # Will have max_val_in_nums elements
    current_prefix_sum = 0
    for g_val in range(1, max_val_in_nums + 1):
        current_prefix_sum += count_gcd_is_g[g_val]
        ps_arr.append(current_prefix_sum) 

    ans = []
    for q_idx in queries:
        # q_idx is a 0-based index into the (conceptual) sorted list of all GCD pair values.
        # We need to find the GCD value `V` that corresponds to this q_idx.
        # The values `V` occupy original indices from `ps_arr[V-2]` to `ps_arr[V-1]-1`.
        # (Where `ps_arr[-1]` is taken as 0).
        # We are looking for the smallest `V` such that `q_idx < ps_arr[V-1]`.
        #
        # `bisect_right(ps_arr, q_idx)` returns a 0-based `insertion_idx` such that:
        #   - all `ps_arr[j] <= q_idx` for `j < insertion_idx`
        #   - all `ps_arr[j] > q_idx` for `j >= insertion_idx`
        # This means `ps_arr[insertion_idx-1] <= q_idx < ps_arr[insertion_idx]`
        # (with `ps_arr[-1]` notionally 0).
        # The `insertion_idx` is `V-1` (0-based index in ps_arr).
        # So the GCD value is `V = insertion_idx + 1`.
        
        insertion_point_idx = bisect_right(ps_arr, q_idx) 
        ans.append(insertion_point_idx + 1)
            
    return ans