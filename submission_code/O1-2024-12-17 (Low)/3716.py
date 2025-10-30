class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        """
        We want the longest subsequence whose consecutive absolute differences
        form a non-increasing sequence. Equivalently, if d_i is the absolute
        difference between consecutive elements in the subsequence, we require:
           d_1 >= d_2 >= d_3 >= ...

        We'll use a dynamic programming approach. Let dp[i][d] be the length of 
        the longest valid subsequence ending at index i with the last absolute 
        difference = d. We'll also maintain prefix_dp[i][d], which is the maximum 
        dp[i][x] among all x >= d.  This lets us quickly compute transitions for 
        non-increasing differences.

        Algorithm (O(n^2) time, which can work for n <= 10^4 with careful implementation):
          1) Initialize dp[i][d] = 1 for all i, d, because by default we can 
             consider (nums[i]) as a 1-length subsequence (though the "difference"
             is only meaningful once we have two or more elements).
          2) For i in [0..n-1]:
               For j in [0..i-1]:
                 d = abs(nums[i] - nums[j])
                 # We can append nums[i] to any subsequence ending at j 
                 # whose last difference ≥ d
                 best_len = prefix_dp[j][d] + 1
                 dp[i][d] = max(dp[i][d], best_len)
          3) After filling dp[i], build prefix_dp[i] in descending order of d:
               prefix_dp[i][299] = dp[i][299]
               for d in [298..0]:
                 prefix_dp[i][d] = max(dp[i][d], prefix_dp[i][d+1])
          4) Track the global maximum dp[i][d] for answer.

        Constraints:
          2 <= len(nums) <= 10^4
          1 <= nums[i] <= 300
        """
        n = len(nums)
        # Maximum absolute difference is at most 299 (since 1 <= nums[i] <= 300)
        MAX_DIFF = 300

        # dp[i][d] = length of longest valid subsequence ending at i with last diff = d
        dp = [[1]*MAX_DIFF for _ in range(n)]
        # prefix_dp[i][d] = max(dp[i][x] for x >= d), to help with "non-increasing" constraint
        prefix_dp = [[1]*MAX_DIFF for _ in range(n)]

        answer = 1  # At least 1 element can always form a (trivial) subsequence

        for i in range(n):
            dp_i = dp[i]
            pref_i = prefix_dp[i]
            # Update dp for pairs (j, i)
            for j in range(i):
                d = abs(nums[i] - nums[j])
                # best possible length if we append nums[i] after any subsequence ending at j
                # whose last difference >= d
                candidate_len = prefix_dp[j][d] + 1
                if candidate_len > dp_i[d]:
                    dp_i[d] = candidate_len
                    if candidate_len > answer:
                        answer = candidate_len
            
            # Build prefix_dp[i] in decreasing order of d
            # prefix_dp[i][MAX_DIFF-1] = dp[i][MAX_DIFF-1]
            pref_i[MAX_DIFF-1] = dp_i[MAX_DIFF-1]
            for diff in range(MAX_DIFF-2, -1, -1):
                pref_i[diff] = max(dp_i[diff], pref_i[diff+1])

        return answer