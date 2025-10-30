from typing import List

# Fenwick Tree (Binary Indexed Tree) for range maximum queries
class FenwickTree:
    def __init__(self, size: int):
        self.size = size
        # Initialize with negative infinity, as sums can be negative.
        # The tree is 1-indexed internally.
        self.tree = [float('-inf')] * (size + 1)

    def update(self, idx: int, val: float):
        # idx is 1-based rank
        while idx <= self.size:
            self.tree[idx] = max(self.tree[idx], val)
            idx += idx & (-idx)  # Move to the next relevant index to update

    def query(self, idx: int) -> float:
        # idx is 1-based rank
        # Query returns the maximum value in the prefix [1, idx]
        max_val = float('-inf')
        while idx > 0:
            max_val = max(max_val, self.tree[idx])
            idx -= idx & (-idx)  # Move to the previous relevant index
        return max_val

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Constraints: 1 <= nums.length <= 10^5. So n >= 1.

        # Step 1: Transform nums[i] to b_arr[i] = nums[i] - i
        b_arr = [nums[i] - i for i in range(n)]

        # Step 2: Coordinate compression for b_arr values
        # Get unique sorted values of b_arr to map them to ranks
        unique_sorted_b = sorted(list(set(b_arr)))
        
        # Map values in b_arr to 1-based ranks
        # ranks_count is M, the number of unique values in b_arr
        ranks_count = len(unique_sorted_b)
        rank_map = {val: i + 1 for i, val in enumerate(unique_sorted_b)}

        # Step 3: Initialize Fenwick tree
        ft = FenwickTree(ranks_count)
        
        # Step 4: Initialize overall maximum sum
        max_overall_sum = float('-inf')
        
        # Step 5: Iterate through the numbers
        for i in range(n):
            current_num_val = nums[i]
            # b_val is nums[i] - i
            current_b_val = b_arr[i] 
            
            # Get the 1-based rank of current_b_val
            rank_of_current_b = rank_map[current_b_val]
            
            # Query Fenwick tree for max_prev_dp for subsequences ending with b_val <= current_b_val
            # This retrieves max(dp[j]) over j < i such that b_arr[j] <= current_b_val
            max_prev_dp = ft.query(rank_of_current_b)
            
            # Calculate current_dp_val: max sum of a balanced subsequence ending with nums[i]
            # If max_prev_dp is -inf (no suitable predecessor found) or non-positive,
            # it's better to start a new subsequence with just nums[i].
            # current_num_val + max(0, max_prev_dp) handles this:
            # - If max_prev_dp = -inf, max(0, -inf) = 0. dp_val = current_num_val.
            # - If max_prev_dp = -5 (e.g.), max(0, -5) = 0. dp_val = current_num_val.
            # - If max_prev_dp = 5 (e.g.), max(0, 5) = 5. dp_val = current_num_val + 5.
            current_dp_val = current_num_val + max(0, max_prev_dp)
            
            # Update Fenwick tree with current_dp_val at the rank of current_b_val
            ft.update(rank_of_current_b, current_dp_val)
            
            # Update the overall maximum sum found so far
            max_overall_sum = max(max_overall_sum, current_dp_val)
            
        # Step 6: Return the result
        # Since n >= 1, max_overall_sum will be a finite number.
        # Sums of integers are integers.
        return int(max_overall_sum)