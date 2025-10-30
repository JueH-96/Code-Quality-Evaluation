class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        """
        Dynamic-programming solution.
        dp[i]  = minimum number of balanced substrings that cover
                 the prefix s[0 : i]   (0 ≤ i ≤ n, dp[0] = 0)
        For every i we try all possible previous cut positions j (0 ≤ j < i).
        If the substring s[j : i] is balanced we can update
                 dp[i] = min(dp[i] , dp[j] + 1)
        A substring is balanced when every character that appears in it
        appears exactly the same number of times.
        """
        n = len(s)
        # prefix[k][i] – number of occurrences of the k-th letter in s[0:i]
        prefix = [[0] * 26]
        for ch in s:
            row = prefix[-1].copy()
            row[ord(ch) - 97] += 1
            prefix.append(row)

        def balanced(j: int, i: int) -> bool:
            """return True iff s[j:i] is balanced"""
            first = 0          # frequency that the first seen character has
            for k in range(26):
                cnt = prefix[i][k] - prefix[j][k]
                if cnt:         # the letter occurs in the substring
                    if first == 0:
                        first = cnt
                    elif cnt != first:
                        return False
            return True         # every non-zero count equals `first`

        INF = n + 1
        dp = [INF] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i - 1, -1, -1):
                if balanced(j, i):
                    dp[i] = min(dp[i], dp[j] + 1)
                    # we cannot do better than 1 for this i
                    if dp[i] == 1:
                        break

        return dp[n]