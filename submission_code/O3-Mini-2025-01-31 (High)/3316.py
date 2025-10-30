from bisect import bisect_right
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # The maximum difference among any two numbers.
        max_diff = nums[-1] - nums[0]
        if max_diff == 0:
            return 0  # all numbers equal => every subsequence’s power is 0

        # Collect differences from every pair.
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                # Because nums is sorted, this is nonnegative.
                diffs.add(nums[j] - nums[i])
        
        # The idea:
        # For a chosen subsequence, its "power" (min gap) is determined by one of the adjacent gaps.
        # Also note that for any fixed integer X, let f(X) be the number of subsequences (of length k)
        # whose every adjacent difference (when sorted) is at least X. Then for any subsequence with power p,
        # it has p >= X for all X = 1,...,p and fails for X = p+1.
        # Hence, summing the power over all subsequences is equal to sum_{X>=1} [# subsequences with p >= X] = sum_{X=1}^{max_diff} f(X)
        #
        # f(X) is monotonic and only changes when X passes one of the differences found.
        # So we only need to compute f(X) at the “breakpoints”. Specifically, candidates for X are:
        #   1, and every (d + 1) for d in diffs, and also (max_diff + 1) as the right‐endpoint.
        cand_set = set()
        cand_set.add(1)
        for d in diffs:
            cand_set.add(d + 1)
        cand_set.add(max_diff + 1)
        cands = sorted(cand_set)
        # Now cands is a sorted list of candidate X-values.
        # For any integer X with cands[i] <= X < cands[i+1], f(X) is constant (since all comparisons
        # "nums[j]-nums[i] >= X" flip exactly when X passes a value d+1).
        
        # We write a function that “counts” valid subsequences when the gap threshold is x,
        # i.e. it computes f(x) = (# subsequences (of length k) with every adjacent gap >= x).
        #
        # We use an iterative dynamic programming approach.
        # In a sorted nums list any k-element subsequence (in increasing order) has power defined
        # as the minimum of differences nums[i+1] - nums[i]. The condition "nums[j]-nums[i] >= x"
        # must hold for every adjacent pair.
        #
        # We define dp[r][j] = number of ways to form a subsequence of length r ending at index j.
        # For r = 1, dp[1][j] = 1 for all j.
        # For r >= 2, dp[r][j] = sum{ dp[r-1][i] for i in range(j) if nums[j]-nums[i] >= x }.
        #
        # To speed up the inner loop we precompute for each j the number of indices i with
        # nums[i] <= nums[j]-x. (That is exactly what bisect_right does.)
        def count_valid(x: int) -> int:
            # Precompute for each index j the count of indices i (with i < j)
            # for which nums[i] <= nums[j] - x.
            pos = [bisect_right(nums, nums[j] - x) for j in range(n)]
            # For subsequences of length 1, each index j gives one way.
            dp = [1] * n
            # Iteratively build subsequences of length 2,3,...,k.
            for r in range(2, k + 1):
                new_dp = [0] * n
                # Build prefix sums for dp so that for an index j we can quickly sum dp[0 ... pos[j]-1].
                prefix = [0] * (n + 1)
                for i in range(n):
                    prefix[i+1] = prefix[i] + dp[i]
                for j in range(n):
                    # Only indices i with i < j and satisfying nums[i] <= nums[j]-x are allowed.
                    # They are exactly the first pos[j] indices.
                    new_dp[j] = prefix[pos[j]]
                dp = new_dp
            return sum(dp)
        
        # Sum results over intervals where f(x) is constant.
        ans = 0
        # For X in [cands[i], cands[i+1]), f(x) is constant and equal to count_valid(cands[i]).
        for i in range(len(cands) - 1):
            L = cands[i]
            R = cands[i+1]
            # f(L) = number of subsequences with power >= L
            cnt = count_valid(L)
            interval_length = R - L  # number of integer values X s.t. L <= X < R.
            ans = (ans + cnt * interval_length) % mod
        return ans % mod