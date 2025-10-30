class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        """
        We want the sum of 'powers' of all subsequences of length k, where
        the 'power' of a subsequence is the minimum absolute difference
        between any two elements in that subsequence.

        Approach:
        1) Sort nums.
        2) Observe that the "minimum absolute difference" in a subsequence
           must be at least some value d if and only if every consecutive pair
           (in sorted order of that subsequence) differs by at least d.
        3) Let all_possible_diffs = sorted list of unique differences a[j]-a[i]
           for all i<j in sorted nums (including 0, to handle duplicates properly).
        4) Define ways(d) as the number of subsequences of length k whose
           pairwise consecutive differences are all >= d (thus min difference >= d).
        5) If dset = [d0, d1, ..., dM-1] in ascending order, note that:
           - The number of subsequences whose min difference is at least d0 = ways(d0).
           - The number of subsequences whose min difference is at least d1 = ways(d1).
           - ...
           - The number of subsequences whose min difference is at least dM-1 = ways(dM-1).
        6) The number of subsequences whose min difference is exactly d_i
           is ways(d_i) - ways(d_{i+1}), for 0 <= i < M-1,
           and ways(d_{M-1}) for the largest d_{M-1}.
        7) Hence the sum of all min differences = 
           sum_{i=0..M-2} d_i * (ways(d_i) - ways(d_{i+1})) 
           + d_{M-1} * ways(d_{M-1}).
        8) Compute ways(d) via a DP that counts how many ways we can pick
           subsequences of length k subject to consecutive gap >= d.

        Since n <= 50, the number of unique differences is at most n*(n-1)/2 = 1225.
        For each difference d, we do a top-down DP with memo of size ~ (n+1)*(n+1)*(k+1).
        Though rather large, careful Python implementation with memoization and pruning
        can pass within time limits.

        We return the result mod 10^9+7.
        """

        import sys
        sys.setrecursionlimit(10**7)
        MOD = 10**9 + 7

        nums_sorted = sorted(nums)
        n = len(nums)

        # Collect unique differences (including 0 to account for duplicates).
        diffs_set = set()
        diffs_set.add(0)
        for i in range(n):
            for j in range(i+1, n):
                diffs_set.add(nums_sorted[j] - nums_sorted[i])
        dset = sorted(diffs_set)  # ascending order of differences

        # Memo for ways(d). We'll cache results so we don't recalc for each d.
        # ways(d) = number of subsequences of length k where consecutive chosen
        # elements differ by at least d (in the sorted array).
        # We'll implement a top-down DP with (last_idx, pos, left) state:
        #   last_idx = index of the last-chosen element in nums_sorted, or -1 if none chosen yet
        #   pos = current index in [0..n] we are deciding to pick or skip
        #   left = how many elements left to pick
        # Then ways_for_d(-1, 0, k) is the total count we want.

        from functools import lru_cache

        def ways_for_d(d):
            @lru_cache(None)
            def dp(last, pos, left):
                if left == 0:
                    return 1
                if pos == n:
                    return 0
                # skip current element
                ans = dp(last, pos+1, left)
                # try picking current element if it respects the gap
                if last == -1 or (nums_sorted[pos] - nums_sorted[last] >= d):
                    ans = (ans + dp(pos, pos+1, left-1)) % MOD
                return ans

            return dp(-1, 0, k)

        # Precompute ways(d) for each d in dset
        ways_array = []
        for d in dset:
            ways_array.append(ways_for_d(d))

        # Now compute sum of min differences using the "distribution" idea:
        # sum_{i=0..m-2} dset[i]*(ways(dset[i]) - ways(dset[i+1])) + dset[m-1]*ways(dset[m-1])
        m = len(dset)
        total_sum = 0
        for i in range(m-1):
            count_exact = (ways_array[i] - ways_array[i+1]) % MOD
            total_sum = (total_sum + dset[i] * count_exact) % MOD
        # Add the last difference's contribution
        total_sum = (total_sum + dset[m-1] * ways_array[m-1]) % MOD

        return total_sum % MOD