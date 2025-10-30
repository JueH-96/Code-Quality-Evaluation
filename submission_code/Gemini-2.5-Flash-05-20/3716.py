import collections

class FenwickTree:
    def __init__(self, size):
        # The Fenwick Tree will store maximum values for a given index.
        # `size` is the maximum possible index (e.g., 299 for MAX_DIFF) + 1.
        # The tree array is 1-indexed internally.
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx, val):
        # Updates the value at `idx` to be the maximum of its current value and `val`.
        # `idx` is 0-based externally, converted to 1-based internally.
        idx += 1 
        while idx <= self.size:
            self.tree[idx] = max(self.tree[idx], val)
            idx += (idx & -idx) # Move to the next relevant node

    def query(self, idx):
        # Queries the maximum value in the range [0, idx] (inclusive).
        # `idx` is 0-based externally, converted to 1-based internally.
        idx += 1
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= (idx & -idx) # Move to the parent node
        return res


class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        MAX_VAL = 300  # Maximum possible value for nums[i]
        MAX_DIFF = 299 # Maximum possible absolute difference (e.g., |300 - 1|)

        # dp_fts[value] will be a FenwickTree storing max lengths for subsequences
        # ending with 'value', indexed by MAPPED_DIFFERENCE.
        # Mapped difference: original_diff -> MAX_DIFF - original_diff.
        # This mapping allows suffix range queries (original_diff >= current_diff)
        # to be performed as prefix range queries (mapped_diff <= MAX_DIFF - current_diff)
        # on the Fenwick Tree.
        dp_fts = [FenwickTree(MAX_DIFF + 1) for _ in range(MAX_VAL + 1)]

        # The minimum length of a valid subsequence is 1 (any single element).
        overall_max_len = 1 

        # To ensure that 'prev_val' truly appeared before 'current_num' in nums,
        # we use a set to track values seen so far.
        seen_values = set() 

        for current_num in nums:
            current_val = current_num
            
            # current_best_for_val[diff] will temporarily store the maximum length
            # of a subsequence ending with 'current_val', where the last difference was 'diff'.
            # This array accumulates results from all possible previous values.
            current_best_for_val = [0] * (MAX_DIFF + 1)

            # Iterate through all possible integer values that could have been the 'prev_val'.
            for prev_val in range(1, MAX_VAL + 1):
                current_diff = abs(current_val - prev_val)
                
                # Calculate the mapped index for the Fenwick Tree query.
                # A query for original_diff >= current_diff becomes a query for mapped_diff <= MAX_DIFF - current_diff.
                mapped_diff_for_query = MAX_DIFF - current_diff
                
                # Query the Fenwick Tree for `prev_val` to get the maximum length of a subsequence
                # ending with `prev_val`, whose last absolute difference was >= `current_diff`.
                len_from_prev_val = dp_fts[prev_val].query(mapped_diff_for_query)

                if len_from_prev_val > 0:
                    # If `len_from_prev_val` is > 0, it means we found a valid subsequence
                    # ending at `prev_val` that can be extended with `current_val`.
                    new_len = len_from_prev_val + 1
                    current_best_for_val[current_diff] = max(current_best_for_val[current_diff], new_len)
                elif prev_val in seen_values:
                    # If `len_from_prev_val` is 0 but `prev_val` has been seen before:
                    # This means `prev_val` either existed only as a single element in previous `nums`,
                    # or any longer subsequence ending in `prev_val` did not satisfy the diff condition
                    # to be extended with `current_val`.
                    # In this scenario, we can always form a new subsequence of length 2: `[prev_val, current_val]`.
                    current_best_for_val[current_diff] = max(current_best_for_val[current_diff], 2)
            
            # After checking all possible `prev_val` for the `current_val`,
            # update the Fenwick Tree for `current_val` with the best lengths found.
            # Also update the `overall_max_len`.
            for diff_val in range(MAX_DIFF + 1):
                if current_best_for_val[diff_val] > 0:
                    # Calculate the mapped index for the Fenwick Tree update.
                    mapped_diff_for_update = MAX_DIFF - diff_val
                    dp_fts[current_val].update(mapped_diff_for_update, current_best_for_val[diff_val])
                    
                    # Update the overall maximum length found so far.
                    overall_max_len = max(overall_max_len, current_best_for_val[diff_val])
            
            # Add `current_val` to the set of seen values.
            seen_values.add(current_val)
        
        return overall_max_len