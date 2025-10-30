class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        INF = float('inf')
        dp = [[INF]*27 for _ in range(n+1)]
        dp[0][0] = 0  # At position 0, with no previous character, cost is 0

        # Function to check if two characters are 'good' (not almost-equal)
        def is_good(prev, c):
            if prev == 0:
                return True
            return c != prev and abs(c - prev) != 1

        for i in range(n):
            for prev in range(27):  # prev characters, 0 means no previous character
                if dp[i][prev] < INF:
                    for c in range(1, 27):  # Possible current characters 'a'(1) to 'z'(26)
                        if is_good(prev, c):
                            cost = dp[i][prev]
                            curr_char = ord(word[i]) - ord('a') + 1
                            if c != curr_char:
                                cost += 1  # Increment cost if we change the character
                            dp[i+1][c] = min(dp[i+1][c], cost)

        result = min(dp[n][1:27])  # Minimum cost among all possible ending characters
        return result