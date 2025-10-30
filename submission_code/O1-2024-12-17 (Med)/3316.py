class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # ---------------------------------------------------------
        # 1) Collect all distinct nonnegative pairwise differences
        #    Also ensure 0 is included, because the min-diff can be 0
        #    in case duplicates exist.
        # ---------------------------------------------------------
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(nums[j] - nums[i])
        diffs.add(0)
        sorted_diffs = sorted(diffs)
        
        # Append a "larger than all" difference so that for
        # the last distinct difference d_M, we can do:
        #   count(d_M) - count(d_{M+1})  (where count(d_{M+1}) = 0)
        # This helps to bucket the exact min-diff = d_M.
        sorted_diffs.append(sorted_diffs[-1] + 1)
        
        # ---------------------------------------------------------
        # 2) Define a function count_subsequences(d) that returns
        #    how many subsequences of length k have min pairwise
        #    difference >= d.  We use dynamic programming over the
        #    sorted array.
        #
        #    We let dp[c][l] = number of ways to pick c elements
        #    from the (current to end) portion of the array if the
        #    last chosen index is l, or l == n if none chosen yet.
        #
        #    We'll iterate i from right to left to fill a new_dp
        #    from dp at each step.
        # ---------------------------------------------------------
        def count_subsequences(d: int) -> int:
            # old_dp[c][l]: ways to pick c elements from the subarray i..n-1
            # if the "last picked index" is l (or l == n for "none so far").
            old_dp = [[0] * (n + 1) for _ in range(k + 1)]
            
            # Base case: picking 0 elements is always 1 way, regardless of l
            for l in range(n + 1):
                old_dp[0][l] = 1
            
            # Build from right to left
            for i in reversed(range(n)):
                new_dp = [row[:] for row in old_dp]  # copy old_dp
                for c in range(1, k + 1):
                    for l in range(n + 1):
                        # We can choose nums[i] if it meets difference requirement
                        if l == n or (nums[i] - nums[l] >= d):
                            new_dp[c][l] = (new_dp[c][l] + old_dp[c - 1][i]) % MOD
                old_dp = new_dp
            
            # The answer is the number of ways to pick k elements starting
            # from index 0..n-1 with "no last chosen" => last == n
            return old_dp[k][n]
        
        # ---------------------------------------------------------
        # 3) Precompute how many k-subsequences have min-diff >= d
        #    for each d in sorted_diffs.
        # ---------------------------------------------------------
        counts = []
        for d in sorted_diffs:
            counts.append(count_subsequences(d))
        
        # ---------------------------------------------------------
        # 4) Use the standard "bucket" trick:
        #    minDiff(S) = d_i exactly if S is counted in count(d_i)
        #    but not in count(d_{i+1}).
        #    Then add d_i * [count(d_i) - count(d_{i+1})].
        # ---------------------------------------------------------
        ans = 0
        for i in range(len(sorted_diffs) - 1):
            diff_here = sorted_diffs[i]
            count_here = counts[i]
            count_next = counts[i + 1]
            ways_exactly = (count_here - count_next) % MOD
            ans = (ans + diff_here * ways_exactly) % MOD
        
        return ans