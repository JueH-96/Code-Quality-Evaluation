from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        num_queries = len(queries)

        # Edge case: If nums is already a Zero Array
        if all(x == 0 for x in nums):
            return 0

        def can_make_zero(k: int, original_nums: List[int], all_queries: List[List[int]]) -> bool:
            """
            Checks if applying the first k queries (from all_queries) can make
            the original_nums array a Zero Array.
            For each index j, this is possible if the target value original_nums[j]
            can be formed as a sum of a subset of relevant query values from the
            first k queries.
            """
            n_check = len(original_nums)

            for j in range(n_check):
                target = original_nums[j]
                if target == 0:
                    continue # Already zero, no requirement for this index

                # Collect relevant query values for index j among the first k queries
                # A query [l, r, val] is relevant to index j if l <= j <= r.
                # These relevant val_i values are the 'items' for the subset sum problem
                # for index j.
                vals_j = []
                for i in range(k): # Iterate through the first k queries
                    l, r, val = all_queries[i]
                    if l <= j <= r:
                        vals_j.append(val)

                # Subset sum DP: Can target be formed using a subset of vals_j?
                # dp[s] is True if sum s is possible using a subset of values in vals_j.
                # The maximum possible target for an index is 1000. The DP table size
                # is at most 1001.
                dp = [False] * (target + 1)
                dp[0] = True # Sum 0 is always possible (by selecting an empty subset)

                # Build the DP table using the available values for index j.
                # This standard 0/1 knapsack DP update iterates through each value 'v'
                # available for this index, and for each value, it updates the dp
                # table considering sums from 'target' down to 'v'. This ensures
                # that each value 'v' from the list `vals_j` is treated as a distinct
                # item that can be used at most once in the subset sum calculation
                # for this specific index and target. This is correct for subset sum
                # on a multiset.
                for v in vals_j:
                    # Iterate sums s from target down to v.
                    # If sum s-v was possible using a subset of values considered so far
                    # (excluding the current value v), then sum s is possible by adding value v.
                    for s in range(target, v - 1, -1):
                        dp[s] = dp[s] or dp[s - v]

                # After considering all relevant query values for index j among
                # the first k queries, check if the target sum is possible.
                if not dp[target]:
                    # If the target sum is not possible for this index,
                    # then these k queries are not sufficient to make nums[j] zero.
                    # Thus, it's not possible to make the entire array zero with k queries.
                    return False

            # If the loop finishes, it means all indices can potentially be made zero
            # using the first k queries.
            return True

        # Binary search loop to find the minimum k.
        # We are searching for the smallest k in the range [1, num_queries]
        # such that can_make_zero(k, ...) returns True.
        low = 1
        high = num_queries
        ans = -1 # Initialize ans to -1, meaning no solution found yet.
                 # If the binary search completes without finding a k that works,
                 # ans will remain -1, correctly indicating impossibility.

        # The binary search works by finding the minimum 'mid' value for which
        # the condition `can_make_zero(mid, ...)` is True.
        while low <= high:
            mid = low + (high - low) // 2
            if can_make_zero(mid, nums, queries):
                # If it's possible to make the array zero using the first 'mid' queries,
                # then 'mid' is a possible answer. We store 'mid' as a potential answer
                # and try if a smaller number of queries (in the left half [low, mid-1])
                # can also achieve the zero array.
                ans = mid
                high = mid - 1 # Shrink the search space to the left
            else:
                # If it's not possible to make the array zero using the first 'mid' queries,
                # then we need more queries. We search in the right half [mid + 1, high].
                low = mid + 1 # Shrink the search space to the right

        # If the loop finishes, 'ans' holds the smallest k found that works.
        # If no such k was found in the range [1, num_queries] (which means
        # even with all num_queries it's not possible), 'ans' remains -1.
        return ans