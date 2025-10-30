class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Build prefix frequency array
        # freq[i][c] = number of times character c appears in s[:i]
        freq = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            for c in range(26):
                freq[i + 1][c] = freq[i][c]
            freq[i + 1][ord(s[i]) - ord('a')] += 1
        
        # dp[i] = minimum number of balanced substrings partitioning s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                # Check if s[j:i] is balanced
                # We'll gather non-zero counts and see if they're all the same
                first_count = None
                all_equal = True
                found_any = False
                for c in range(26):
                    count_in_sub = freq[i][c] - freq[j][c]
                    if count_in_sub > 0:
                        found_any = True
                        if first_count is None:
                            first_count = count_in_sub
                        elif count_in_sub != first_count:
                            all_equal = False
                            break
                if found_any and all_equal:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]