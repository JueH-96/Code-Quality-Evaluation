class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        """
        Dynamic programming.
        dp[i][c] – minimal number of changes needed for the prefix that ends at
        position i (0-indexed) if we finally put character with code c
        (0 … 25) at position i.

        Transition:
            dp[i][c] = min_{p : |p-c| >= 2}  dp[i-1][p] + cost
        where   cost = 0  if original character already is c
                     = 1  otherwise.

        The condition |p-c| >= 2 guarantees that the two adjacent characters
        are NOT almost-equal (they differ by **at least** 2 in the alphabet).

        Complexity:  O(n * 26 * 26) ≤ 100 * 676 = 67 600   – completely fine.
        """
        n = len(word)
        # numeric codes for the original word
        codes = [ord(ch) - 97 for ch in word]
        
        INF = 10**9
        # dp for position 0
        dp = [0 if c == codes[0] else 1 for c in range(26)]
        
        # iterate through the rest of the string
        for i in range(1, n):
            ndp = [INF] * 26
            for cur in range(26):
                cost = 0 if cur == codes[i] else 1
                best_prev = INF
                for prev in range(26):
                    if abs(cur - prev) >= 2:            # not almost-equal
                        best_prev = min(best_prev, dp[prev])
                ndp[cur] = best_prev + cost
            dp = ndp
        
        return min(dp)