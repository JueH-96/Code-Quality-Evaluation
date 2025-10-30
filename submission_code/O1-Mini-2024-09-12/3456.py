class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        dp = [defaultdict(int) for _ in range(k+1)]

        for num in nums:
            # Iterate in reverse to prevent overwriting
            for c in range(k, -1, -1):
                for last_num, length in list(dp[c].items()):
                    if num == last_num:
                        dp[c][num] = max(dp[c][num], length + 1)
                    else:
                        if c + 1 <= k:
                            dp[c + 1][num] = max(dp[c + 1][num], length + 1)
            # Start a new subsequence with num
            dp[0][num] = max(dp[0][num], 1)

        # Get the maximum length from all dp
        return max(max(d.values()) for d in dp if d)