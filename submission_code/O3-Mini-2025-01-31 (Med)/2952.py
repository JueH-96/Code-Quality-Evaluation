from typing import List
import math

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Check time T = 0 (i.e. no seconds, no increments)
        s1 = sum(nums1)
        if s1 <= x:
            return 0

        # We'll use a dynamic programming approach for the case T < n.
        # For each candidate T seconds, we are allowed to "effectively" reset at most T indices
        # (each index can be reset at most once; and resets must come at different seconds).
        # The effect of a reset on index i if assigned at time t (1 <= t <= T) is that the final value becomes:
        #    (T - t)*nums2[i]  rather than  nums1[i] + T*nums2[i].
        # Thus, the "saving" (benefit) for resetting index i at time t is:
        #    (nums1[i] + T*nums2[i]) - ((T - t)*nums2[i]) = nums1[i] + t*nums2[i]
        # If we choose to reset a set S of indices (|S| = k) with k ≤ min(n, T),
        # and we can assign them distinct times, then the total saving is
        #    sum_{i in S} [nums1[i] + (assigned reset time)*nums2[i] ]
        # and we want to maximize this saving.
        #
        # Note: When T >= n, we can reset all indices.
        # Then by giving the largest available reset time to the index with the largest nums2,
        # the maximum total saving is:
        #    sum(nums1) + sum_{j=1}^{n} ((T - j + 1) * sorted_desc[j])
        # In that case the final sum would be:
        #    final_sum = (sum(nums1) + T*sum(nums2)) - { sum(nums1) + sum_{j=1}^{n} ((T - j + 1) * sorted_desc[j]) }
        #              = T*sum(nums2) - sum_{j=1}^{n} ((T - j + 1) * sorted_desc[j])
        # A short algebra shows that for T >= n, the final sum does not depend on T.
        #
        # For T < n, we need to choose exactly T indices to reset optimally.
        # We can precompute a DP table over all indices after sorting by nums2 in an order
        # that makes the optimal assignment straightforward.
        #
        # It turns out that if we sort the indices by nums2 in non-decreasing order, then
        # the optimal assignment (which gives each chosen index a multiplier equal to its order of selection)
        # is given by:
        #    dp[i][k] = maximum total saving we can get by considering the first i indices
        #               and selecting exactly k indices to reset, where if we take an index,
        #               its contribution is: nums1 + (k)*nums2 (that is, if it is the k-th chosen).
        #
        # The recurrence is:
        #    dp[i][k] = max( dp[i-1][k],  dp[i-1][k-1] + nums1[i-1] + k * nums2[i-1] )
        #
        # Once the DP is computed, for a candidate time T (with T < n),
        # the best total saving for T resets is dp[n][T].
        # Meanwhile, if we do no reset on an index, its final value is: nums1[i] + T*nums2[i]
        # so the total sum without resets is:
        #    base = sum(nums1) + T * sum(nums2)
        # And then by applying resets on T indices, the final sum becomes:
        #    final_sum = base - dp[n][T]
        #
        # We want final_sum <= x.
        #
        # Finally, note that for any T >= n, we can always use all n resets.
        # In that case the maximum saving becomes (with optimal assignment, assigning the available
        # reset times T-n+1, T-n+2, ... , T to the indices with the largest nums2):
        #    saving_all = sum(nums1) + sum_{j=1}^{n} ((T - n + j) * sorted_desc[j])
        # and a simple algebra shows that the final sum is constant:
        #    final_sum = sum_{j=1}^{n} ((j - 1)* sorted_desc[j])
        #
        # We will compute the DP for choices of k = 0 to n.
        #
        # Let's sort the indices by nums2 in non-decreasing order.
        arr = list(zip(nums1, nums2))
        arr.sort(key=lambda x: x[1])
        # Precompute DP table: dp[i][k] for i = 0..n, k = 0..n.
        # dp[0][0] = 0, and dp[0][k] = -infinity for k>=1.
        dp = [[-10**9 for _ in range(n+1)] for _ in range(n+1)]
        dp[0][0] = 0
        for i in range(1, n+1):
            a, b = arr[i-1]
            for k in range(0, i+1):
                # option not to take the i-th element
                dp[i][k] = dp[i-1][k]
                if k > 0:
                    # if we take the current element as the k-th reset,
                    # its contribution is: a + k * b
                    candidate = dp[i-1][k-1] + a + k * b
                    if candidate > dp[i][k]:
                        dp[i][k] = candidate

        total_nums1 = sum(nums1)
        total_nums2 = sum(nums2)
        ans = -1

        # Search candidate times T from 0 to n-1.
        # (For T=0, no resets: final sum = sum(nums1)).
        for T in range(0, n):
            base = total_nums1 + T * total_nums2
            best_saving = dp[n][T]  # best total saving using exactly T resets
            final_sum = base - best_saving
            if final_sum <= x:
                ans = T
                return ans  # since we are iterating in increasing T

        # Now consider the case when T >= n.
        # When T >= n, we can reset every index. In that optimum strategy,
        # we assign reset times to all indices optimally.
        # It can be shown that the final sum is independent of T and equals:
        #   final_sum_all = sum_{j=1}^{n} ((j - 1) * sorted_desc[j])
        # where sorted_desc is the list of nums2 values sorted in descending order.
        sorted_desc = sorted(nums2, reverse=True)
        final_sum_all = 0
        for j in range(1, n+1):
            final_sum_all += (j - 1) * sorted_desc[j-1]
        if final_sum_all <= x:
            # The minimal T in the T >= n range is n
            return n
        else:
            return -1