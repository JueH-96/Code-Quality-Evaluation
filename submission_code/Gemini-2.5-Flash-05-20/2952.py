from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)

        # Calculate initial sum of nums1 and nums2 elements.
        # These represent the total accumulated values if no operations are performed.
        S1 = sum(nums1)
        S2 = sum(nums2)

        # Base case: If the initial sum of nums1 is already less than or equal to x,
        # then no time (0 seconds) is needed.
        if S1 <= x:
            return 0

        # Create pairs (nums1[i], nums2[i]).
        # Sort these pairs based on nums2[i] in ascending order.
        # This sorting is crucial for the DP state transition.
        # If nums2[i] values are equal, the relative order of nums1[i] values doesn't affect the maximum sum calculation for reduction,
        # so Python's stable sort or an arbitrary secondary key (like nums1[i] itself) is fine.
        pairs = sorted([(nums1[i], nums2[i]) for i in range(n)], key=lambda p: p[1])

        # dp[j] will store the maximum possible reduction in the total sum if we perform exactly `j` operations.
        # The reduction for an element (s1, s2) reset at the k-th 'opportunity' is s1 + k * s2.
        # By sorting pairs by s2 ascending, when we select j items and assign them logical reset times (or ranks) 1 to j,
        # the element with the k-th smallest s2 among the chosen j items is paired with rank k.
        # The `j` in `j * s2` in the DP update represents this rank.
        # dp is of size n+1 to account for 0 to n operations.
        dp = [0] * (n + 1)

        # Iterate through each item (s1, s2) in the sorted pairs.
        for s1, s2 in pairs:
            # Iterate `j` backwards from `n` down to `1`.
            # This ensures that for `dp[j]`, the current `(s1, s2)` pair is considered as
            # the `j`-th item (in terms of sorted nums2 value) among the `j` items selected for this `dp` state.
            # Also, iterating backwards prevents using the same item multiple times in the formation of `dp[j]` from `dp[j-1]`
            # (which is characteristic of 0/1 knapsack problems).
            for j in range(n, 0, -1):
                # Update dp[j]:
                # We can either not include the current (s1, s2) item in our set of `j` operations (keeping dp[j] as is),
                # OR include it. If we include it, it implies we are forming a set of `j` operations,
                # and this (s1, s2) item is the `j`-th element when ordered by `s2` value.
                # So its contribution to the reduction would be `s1 + j * s2`, added to the maximum reduction
                # achievable with `j-1` operations (dp[j-1]).
                dp[j] = max(dp[j], dp[j-1] + s1 + j * s2)
        
        # Now, dp[k] for k from 0 to n contains the maximum possible reduction by making k operations.
        # We need to find the minimum time 't' such that the total sum of nums1 becomes less than or equal to x.
        # At time `t`, the total sum if no operations were performed would be `S1 + t * S2`.
        # We can perform at most `t` operations (one per second), and at most `n` operations (since there are only n elements).
        # So, we can perform `k = min(t, n)` operations.
        # Since `nums1[i]` and `nums2[i]` are non-negative, the reduction `s1 + j * s2` is always non-negative.
        # This implies that `dp[k]` is a non-decreasing sequence (dp[k] >= dp[k-1]).
        # Therefore, to minimize the final sum, for any given `t`, we should apply the maximum possible reduction, which is `dp[min(t, n)]`.
        # The condition we are looking for is: `S1 + t * S2 - dp[min(t, n)] <= x`.

        # We already checked t=0. Now, iterate through possible values of t from 1 up to n.
        for t in range(1, n + 1):
            # The current_sum_at_t considers the natural growth and the maximum reduction for 't' operations.
            current_sum = S1 + t * S2 - dp[t]
            if current_sum <= x:
                return t
        
        # If the loop finishes, it means no solution was found for t from 1 to n.
        # Consider cases for t > n:
        # For any t > n, we can still only perform at most 'n' operations, so the maximum reduction remains `dp[n]`.
        # The inequality becomes: `S1 + t * S2 - dp[n] <= x`.

        # Case 1: S2 > 0.
        # The term `t * S2` will increase as `t` increases. The reduction `dp[n]` is constant.
        # So, the expression `S1 + t * S2 - dp[n]` is an increasing function of `t`.
        # If the condition was not met for `t=n` (which was `S1 + n * S2 - dp[n] <= x`),
        # it means `S1 + n * S2 - dp[n] > x`.
        # Since the function is increasing, it will be even greater for any `t > n`.
        # Therefore, if `S2 > 0` and we reach this point, it's impossible to satisfy the condition.

        # Case 2: S2 == 0.
        # The term `t * S2` is always 0. The expression becomes `S1 - dp[n]`. This is a constant value.
        # If `S1 - dp[n] <= x` was true, we would have returned `n` in the loop.
        # If `S1 - dp[n] > x`, it means it's impossible, as the sum cannot be reduced further by more operations or time.
        # Therefore, if `S2 == 0` and we reach this point, it's impossible.

        # In both scenarios (S2 > 0 or S2 == 0), if no solution was found for t up to n, it's impossible.
        return -1