from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        sumA = sum(nums1)
        
        # If the array already sums to x or less, no steps needed.
        if sumA <= x:
            return 0
        
        # Sum of all increments:
        sumB = sum(nums2)
        
        # ----------------------------------------------------------------
        # Special case: if all nums2 are zero, the values in nums1 never grow.
        # We can only set one element to zero per step.  We want to see if
        # we can reduce sumA to <= x by zeroing out some of the largest elements.
        # ----------------------------------------------------------------
        if sumB == 0:
            # Greedy: remove largest nums1 elements first
            # until sumA <= x or we run out of elements.
            arr_desc = sorted(nums1, reverse=True)
            steps = 0
            curr_sum = sumA
            while steps < n and curr_sum > x:
                curr_sum -= arr_desc[steps]
                steps += 1
            return steps if curr_sum <= x else -1
        
        # ----------------------------------------------------------------
        # General case: nums2 has at least one positive element.
        #
        # Each second t, we add nums2[i] to nums1[i], then choose ONE index
        # to reset to 0.  Over T steps, effectively we can "remove" up to T
        # different indices (one per step), but each chosen index i contributes
        # a "chunk" of (nums1[i] + k*nums2[i]) depending on the exact step k
        # it was removed.  The optimal strategy (to maximize sum removed) is
        # to assign bigger multiplier-k to those indices having bigger nums2[i].
        #
        # We will compute, for t = 0..n, the maximum total "chunk" that can be
        # removed by exactly t resets (picking t distinct indices).  Then if T>n,
        # we can pick all n indices and the effective multiplier grows uniformly,
        # yielding a closed-form.  Finally we check if the final sum <= x.
        #
        # DP approach (sort pairs by b ascending) so that picking item j as the
        # t-th largest b contributes (a_j + t*b_j).  Then final sum after t steps
        # = sumA + t*sumB - dp[t][n].
        # ----------------------------------------------------------------
        
        # Pair up (a_i, b_i) and sort by b ascending
        pairs = [(nums1[i], nums2[i]) for i in range(n)]
        pairs.sort(key=lambda x: x[1])  # sort ascending by b
        
        # Build DP table: dp[t][j] = max sum-removal by picking t items out of first j
        # items (in ascending b order).  If we pick the j-th item as the t-th item,
        # we add a_j + t*b_j.
        
        # dp array of size (n+1) x (n+1), initialize to -inf except dp[0][j]=0
        import math
        dp = [[-math.inf]*(n+1) for _ in range(n+1)]
        for j in range(n+1):
            dp[0][j] = 0  # picking 0 items => 0 sum removed
        
        for j in range(1, n+1):
            a_j, b_j = pairs[j-1]
            for t in range(0, j+1):
                # Case 1: do not pick the j-th item
                dp[t][j] = max(dp[t][j], dp[t][j-1])
                # Case 2: pick the j-th item as the t-th chosen
                if t > 0:
                    candidate = dp[t-1][j-1] + a_j + t*b_j
                    if candidate > dp[t][j]:
                        dp[t][j] = candidate
        
        # Try all t in [0..n].  final_sum(t) = sumA + t*sumB - dp[t][n].
        # We want to find the smallest t with final_sum(t) <= x.
        best = None
        for t in range(n+1):
            removed = dp[t][n]           # maximum chunk removed with t resets
            final_sum = sumA + t*sumB - removed
            if final_sum <= x:
                best = t
                break
        
        if best is not None:
            return best
        
        # If we get here, none t in [0..n] worked.  For T > n, the final sum is
        #   sumA + T*sumB - [ dp(n,n) + (T-n)*sum_of_all_b ]
        # but it simplifies to a constant = sumA + n*sumB - dp(n,n).
        # If that constant > x, then -1; else answer is n.
        max_removed_all = dp[n][n]  # pick all n items
        const_sum = sumA + n*sumB - max_removed_all
        if const_sum <= x:
            # any T >= n works, so the minimal T is n
            return n
        
        return -1