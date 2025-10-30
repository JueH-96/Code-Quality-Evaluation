from typing import List, Tuple

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        num_queries = len(queries)
        initial_nums_tuple = tuple(nums) # Convert to tuple for immutability and use as a base

        # Binary search for k
        # k can range from 0 (if nums is already zero) to num_queries
        low = 0
        high = num_queries + 1 # `high` is exclusive, so `num_queries` is included in the search space
        ans = -1

        while low < high:
            mid = low + (high - low) // 2

            if mid == 0:
                # Special case: If k=0, check if nums is already a Zero Array.
                if all(x == 0 for x in initial_nums_tuple):
                    ans = 0
                    high = mid # Try to achieve with even fewer (0) queries
                else:
                    low = mid + 1 # Cannot achieve with 0 queries, need more
                continue

            # Check if `nums` can be made zero using the first `mid` queries.
            # `self._check` method encapsulates the recursive DP logic.
            if self._check(mid, initial_nums_tuple, queries, n):
                ans = mid # `mid` queries are sufficient, try for a smaller k
                high = mid
            else:
                low = mid + 1 # `mid` queries are not sufficient, need more
        
        return ans

    def _check(self, k_val: int, initial_nums_tuple: Tuple[int, ...], queries: List[List[int]], n: int) -> bool:
        # memo stores (query_idx, current_nums_tuple) -> bool
        # This memoization table is specific to each call of _check,
        # ensuring states are correctly tracked for the given k_val.
        memo = {}

        def can_zero_recursive(query_idx: int, current_nums_list: List[int]) -> bool:
            """
            Recursive helper function to determine if `current_nums_list` can be made all zeros
            starting from `query_idx` up to `k_val-1` queries.
            """
            current_nums_tuple = tuple(current_nums_list)
            
            # Memoization check: If this state has been computed, return cached result.
            if (query_idx, current_nums_tuple) in memo:
                return memo[(query_idx, current_nums_tuple)]

            # Base Case 1: All numbers in `current_nums_list` are zero. Success!
            if all(x == 0 for x in current_nums_list):
                return True
            
            # Base Case 2: We have used all `k_val` queries but `nums` is not zero. Failure.
            if query_idx == k_val:
                return False

            # Get the current query's details
            l, r, val = queries[query_idx]

            # Inner recursive function to explore all 2^(r-l+1) ways of applying the current query.
            # It makes decisions for indices from `idx_in_nums` up to `r`.
            def decide_for_indices(idx_in_nums: int, temp_nums_list: List[int]) -> bool:
                """
                Explores options for applying the current query (queries[query_idx])
                to indices from `idx_in_nums` to `r`.
                """
                # Base Case for inner recursion: All decisions for the current query's range [l, r] are made.
                if idx_in_nums > r:
                    # Proceed to the next query with the updated `temp_nums_list`.
                    return can_zero_recursive(query_idx + 1, temp_nums_list)

                # Option 1: Do NOT apply `val` to `nums[idx_in_nums]`.
                # If this path leads to a solution, we take it.
                if decide_for_indices(idx_in_nums + 1, temp_nums_list):
                    return True

                # Option 2: DO apply `val` to `nums[idx_in_nums]`.
                # This is only valid if `nums[idx_in_nums]` does not become negative.
                if temp_nums_list[idx_in_nums] >= val:
                    # Create a new list copy to represent the state if `val` is applied.
                    next_temp_nums_list = list(temp_nums_list) 
                    next_temp_nums_list[idx_in_nums] -= val
                    
                    # If this path leads to a solution, we take it.
                    if decide_for_indices(idx_in_nums + 1, next_temp_nums_list):
                        return True
                
                # If neither option leads to a solution from this `idx_in_nums`, then this path fails.
                return False

        # Start the recursive process for the current `k_val`.
        # Initial call for `can_zero_recursive` starts with the first query (index 0)
        # and a mutable copy of the initial state of `nums`.
        initial_nums_list = list(initial_nums_tuple)
        result = can_zero_recursive(0, initial_nums_list)
        
        # The result of `can_zero_recursive` is the result for `_check`.
        # This result is stored in the `memo` within `can_zero_recursive` itself.
        return result