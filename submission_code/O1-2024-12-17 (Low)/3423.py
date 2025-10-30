class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        We want to handle queries where each query updates nums[pos] = x, and then we must
        compute the maximum sum of a subsequence of nums with no two adjacent elements chosen.

        A classic way to compute this for a static array nums is dynamic programming:
          dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        with base cases dp[0] = max(0, nums[0]) (since we can choose an empty subsequence),
        dp[1] = max(dp[0], nums[1]) if i >= 1, etc.

        The challenge here is we may have up to 5e4 updates (queries), and recomputing dp
        from scratch (O(n)) for each query would be O(n*q) = 2.5e9 in the worst case, which
        is likely too large in a strict time setting. However, a key observation is that
        after updating nums[pos], the dp array only needs to be recomputed from index pos
        onward, and we can stop early if it no longer changes.

        Why can we stop early? dp[i] depends only on dp[i-1], dp[i-2], and nums[i]. If at
        some index i, the newly computed dp[i] matches the old dp[i], then dp[i+1] will
        not change either (since dp[i+1] depends on dp[i], dp[i-1] which are unchanged).
        So we can break out once the dp value remains the same.

        This approach can often pass in practice if the number of indices that actually
        change per query is small on average (even though worst-case complexity can be high).
        We'll implement it carefully.

        We sum up the dp[n-1] values after each query (the "answer" for that query),
        and return the total modulo 10^9+7.
        """

        MOD = 10**9 + 7
        n = len(nums)

        # dp[i] will hold the maximum sum of a non-adjacent subsequence of nums up to index i
        dp = [0]*n

        # Initialize dp array for the original nums
        if n > 0:
            dp[0] = max(0, nums[0])  # we can choose an empty subsequence or nums[0]
        if n > 1:
            dp[1] = max(dp[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])

        ans_sum = 0  # we'll accumulate the answers for each query here

        for pos, x in queries:
            # Update nums[pos]
            nums[pos] = x

            # Recompute dp starting from pos (handle base cases carefully).
            if pos == 0:
                # Recompute dp[0], dp[1], etc.
                dp[0] = max(0, nums[0])
                start_idx = 1
                if n > 1:
                    old_val = dp[1]
                    dp[1] = max(dp[0], nums[1])
                    if dp[1] != old_val:
                        start_idx = 2
            elif pos == 1:
                # Recompute dp[1] using dp[0]
                old_val = dp[1]
                dp[1] = max(dp[0], nums[1])
                if dp[1] != old_val:
                    start_idx = 2
                else:
                    start_idx = 2  # We'll check i=2 below anyway
            else:
                # general case for dp[pos]
                old_val = dp[pos]
                dp[pos] = max(dp[pos-1], dp[pos-2] + nums[pos])
                if dp[pos] != old_val:
                    start_idx = pos + 1
                else:
                    start_idx = pos + 1  # We'll check further if anything might change

            # Propagate changes forward
            for i in range(start_idx, n):
                old_val = dp[i]
                # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
                # Need to handle i-2 carefully if i-2 < 0
                if i == 0:
                    dp[i] = max(0, nums[i])
                elif i == 1:
                    dp[i] = max(dp[0], nums[i])
                else:
                    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
                if dp[i] == old_val:
                    # No further changes will occur beyond this point, so we can stop
                    break

            # Now dp[n-1] is the answer to this query
            ans_sum = (ans_sum + dp[n-1]) % MOD

        return ans_sum % MOD