class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        # We define a DP approach:
        # Let n = len(word).
        # We'll use dp[i][c] to represent the minimum cost (number of changes)
        # to make the substring word[:i+1] valid (no adjacent almost-equal chars),
        # if we fix the character at position i to be 'a' + c.
        #
        # Two letters are 'almost-equal' if they are the same or differ by 1 in ASCII code.
        # Therefore, to avoid a bad pair (c, k), we need:
        #   c != k and abs(c - k) != 1
        #
        # The recurrence:
        #   dp[i][c] = min( dp[i-1][k] ) + (1 if character at i != c else 0)
        #   where the minimum is taken over all k that do not violate the adjacency constraint.
        #
        # We'll fill dp[0][c] first, based on whether word[0] is c or not.
        # Then for each i>0, we compute dp[i][c] from dp[i-1][k].
        
        n = len(word)
        # Edge case: if the string length is 1, no adjacent pairs exist => no operations needed.
        if n == 1:
            return 0
        
        # Convert the word into numerical form for convenience
        # e.g., 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
        numeric_word = [ord(ch) - ord('a') for ch in word]
        
        # Initialize our DP array
        # We'll use a large number (like float('inf')) to represent uninitialized / impossible states
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        # Base case for i = 0
        for c in range(26):
            dp[0][c] = 0 if c == numeric_word[0] else 1
        
        # Fill dp for i = 1..n-1
        for i in range(1, n):
            for c in range(26):
                # Cost to change word[i] to character c
                cost_change = 0 if c == numeric_word[i] else 1
                # We pick the best k from dp[i-1] that doesn't violate adjacency
                best_prev = float('inf')
                for k in range(26):
                    # Avoid adjacent almost-equal: c != k and abs(c - k) != 1
                    if c != k and abs(c - k) != 1:
                        best_prev = min(best_prev, dp[i-1][k])
                dp[i][c] = best_prev + cost_change
        
        # The answer is the minimum among dp[n-1][c] for c in [0..25]
        return min(dp[n-1])