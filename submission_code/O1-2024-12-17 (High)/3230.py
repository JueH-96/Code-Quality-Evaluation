class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n <= 1:
            return 0

        # Convert each character in the word to its index in [0..25].
        letters = [ord(ch) - ord('a') for ch in word]

        # Precompute cost of changing from one letter index to another:
        # cost_change[a][b] = 0 if a == b else 1
        cost_change = [[0 if i == j else 1 for j in range(26)] for i in range(26)]

        # Precompute allowed transitions: we only allow c1->c2 if abs(c1-c2) > 1
        # (meaning they are neither the same nor adjacent in the alphabet).
        allowed = [[True if abs(i - j) > 1 else False for j in range(26)] for i in range(26)]

        # dp[i][c] = minimum number of changes to fix characters up to index i
        #            if we choose character 'c' at position i.
        dp = [[float('inf')] * 26 for _ in range(n)]

        # Initialize dp for i = 0
        for c in range(26):
            dp[0][c] = cost_change[letters[0]][c]

        # Fill dp for 1..n-1
        for i in range(1, n):
            for c in range(26):
                # Cost to change word[i] to character c
                this_cost = cost_change[letters[i]][c]
                # Choose the best c' from previous position
                best_prev = float('inf')
                for prev_c in range(26):
                    if allowed[prev_c][c]:
                        best_prev = min(best_prev, dp[i-1][prev_c])
                dp[i][c] = best_prev + this_cost

        # The answer is the minimum dp value at the last position
        return min(dp[n-1])