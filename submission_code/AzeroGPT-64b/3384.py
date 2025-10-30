class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        dp = [0] * n

        for i in range(n):
            dp[i] = 1 if i%k == 0 and i > 0 else 0  # Base case
            # 'min' function is used to avoid out-of-bound errors
            dp[i] = min(dp[i], dp[i-k] + (word[i] != word[i-k]))

        # The answer is effectively the number of mismatches spread in the entire string divided by k (as successive k-sized blocks can be matched)
        return dp[-1] // k  # OR sum(dp) // k
		# Time: O(n), Space: O(n)