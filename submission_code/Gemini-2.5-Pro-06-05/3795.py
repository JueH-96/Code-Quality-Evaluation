from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        This problem asks for the minimum number of initial queries (k) required to make
        the entire `nums` array zero. The key properties of the problem are:
        1. Monotonicity: If `nums` can be zeroed with `k` queries, it can also be
           zeroed with `k+1` queries. This suggests that binary search on `k` is a
           viable approach.
        2. Independent choices per index: For a given query, the decision to
           decrement `nums[i]` is independent of the decision for `nums[j]` (for i != j).
           This allows us to solve the problem for each `nums[i]` independently and
           combine the results.

        The core of the solution is a function `can_make_zero(k)` which we use
        in the binary search. This function checks if it's possible to zero out `nums`
        using the first `k` queries.

        To implement `can_make_zero(k)`, we check each `nums[i]` one by one. For a
        given `nums[i]`, we need to determine if there's a selection of queries
        (from the first `k` queries) that apply to index `i`, such that their values
        sum up to exactly `nums[i]`. Additionally, we must respect the constraint
        that `nums[i]` never becomes negative during the sequential application of queries.

        This subproblem for each `nums[i]` can be modeled as a dynamic programming
        problem, similar to the 0/1 knapsack or subset sum problem. Let `dp[s]` be a
        boolean indicating whether a sum `s` can be achieved by a valid sequence of
        decrements.
        """
        n = len(nums)
        m = len(queries)

        def can_make_zero(k: int) -> bool:
            for i in range(n):
                target = nums[i]
                if target == 0:
                    continue

                # dp[s] is True if a sum of s is achievable for nums[i] using a valid
                # subset of the first k queries that apply to index i.
                dp = [False] * (target + 1)
                dp[0] = True

                for j in range(k):
                    l, r, v = queries[j]
                    
                    if l <= i <= r:
                        # Iterate s downwards to use query j's value at most once per existing sum.
                        for s in range(target, v - 1, -1):
                            # To achieve sum s from a previous sum (s-v):
                            # 1. It must have been possible to achieve the sum (s-v).
                            # 2. The non-negativity constraint must be met. Before applying
                            #    query j, the value at nums[i] is `target - (s-v)`. This must
                            #    be >= v.
                            if dp[s - v] and (target - (s - v) >= v):
                                dp[s] = True
                
                if not dp[target]:
                    return False
            
            return True

        # Binary search for the minimum k
        low = 1
        high = m
        ans = -1

        while low <= high:
            mid = low + (high - low) // 2
            if can_make_zero(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
            
        return ans