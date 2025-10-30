class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        n = len(s)

        # Helper function to count vowels in a substring
        def count_vowels(sub):
            return sum(1 for char in sub if char in vowels)

        # Dynamic programming table to store results of subproblems
        dp = [[False] * n for _ in range(n)]

        # Fill the dp table
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                substring = s[i:j+1]
                vowel_count = count_vowels(substring)

                if vowel_count % 2 == 1:
                    # Alice's turn
                    dp[i][j] = True
                    for k in range(i, j):
                        if not dp[i][k] or not dp[k+1][j]:
                            dp[i][j] = False
                            break
                else:
                    # Bob's turn
                    dp[i][j] = False
                    for k in range(i, j):
                        if dp[i][k] and dp[k+1][j]:
                            dp[i][j] = True
                            break

        return dp[0][n-1]