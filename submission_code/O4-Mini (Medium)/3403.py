class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # Build prefix sums for counts of each character
        # pref[i][c] = count of char c in s[:i]
        pref = [[0]*26 for _ in range(n+1)]
        for i, ch in enumerate(s, 1):
            c = ord(ch) - ord('a')
            for k in range(26):
                pref[i][k] = pref[i-1][k]
            pref[i][c] += 1

        # dp[i] = minimum number of balanced substrings for s[i:]
        INF = float('inf')
        dp = [INF] * (n+1)
        dp[n] = 0

        # helper to check if s[i:j] is balanced
        def is_balanced(i: int, j: int) -> bool:
            cnts = [pref[j][k] - pref[i][k] for k in range(26)]
            # find the non-zero counts
            non_zero = [c for c in cnts if c > 0]
            if not non_zero:
                return True
            # balanced if all non-zero counts are equal
            first = non_zero[0]
            for c in non_zero:
                if c != first:
                    return False
            return True

        # Fill dp from the end backwards
        for i in range(n-1, -1, -1):
            # We can always take single char
            dp[i] = 1 + dp[i+1]
            # Try longer substrings s[i:j]
            # Early exit if we already found the best possible (1)
            if dp[i] == 1:
                continue
            for j in range(i+2, n+1):
                if is_balanced(i, j):
                    dp[i] = min(dp[i], 1 + dp[j])
                    if dp[i] == 1:
                        break

        return dp[0]