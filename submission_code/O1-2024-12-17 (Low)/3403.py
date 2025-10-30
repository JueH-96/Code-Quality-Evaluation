class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Precompute prefix frequency of each character
        # freq[i][c] will hold the frequency of the character 'c'
        # in the substring s[:i]
        freq = [[0]*26 for _ in range(n+1)]
        for i in range(n):
            for c in range(26):
                freq[i+1][c] = freq[i][c]
            freq[i+1][ord(s[i]) - ord('a')] += 1

        # This function checks if s[start:end] (end non-inclusive) is balanced
        def is_balanced(start, end):
            # Compute frequency of each character in s[start:end]
            # freqInSub[c] = freq[end][c] - freq[start][c]
            freqInSub = [freq[end][c] - freq[start][c] for c in range(26)]
            non_zero = [count for count in freqInSub if count > 0]
            # A substring is balanced if all non-zero frequencies are the same
            return len(non_zero) == 0 or all(x == non_zero[0] for x in non_zero)

        # dp[i] will store the minimum number of balanced substrings
        # partitioning s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n+1):
            # Try all possible starts j of the last balanced substring
            for j in range(i):
                if is_balanced(j, i):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n]