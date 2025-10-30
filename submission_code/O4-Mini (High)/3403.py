class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] = min number of balanced substrings to partition s[:i]
        INF = n + 1
        dp = [INF] * (n + 1)
        dp[0] = 0

        for i in range(n):
            # We'll build substrings ending at i, i.e. s[j:i+1] for j in [0..i]
            freq = [0] * 26
            # freq_count[c] = how many characters in the current window have count == c
            freq_count = {}
            for j in range(i, -1, -1):
                ci = ord(s[j]) - ord('a')
                old = freq[ci]
                new = old + 1
                freq[ci] = new

                # remove one from the old count bucket
                if old > 0:
                    cnt_old = freq_count[old]
                    if cnt_old == 1:
                        del freq_count[old]
                    else:
                        freq_count[old] = cnt_old - 1
                # add one to the new count bucket
                freq_count[new] = freq_count.get(new, 0) + 1

                # if there's exactly one nonzero bucket, it's balanced
                if len(freq_count) == 1:
                    dp[i + 1] = min(dp[i + 1], dp[j] + 1)

        return dp[n]