class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        """
        We want to ensure that no pair of adjacent characters are almost-equal.
        'Almost-equal' means they are the same or differ by 1 in ASCII code.
        We'll use a dynamic programming approach:
        
        dp[i][ch] = the minimum cost (number of changes) to form a valid string
                    ending at index i with character `ch`.
        Transition:
            dp[i][ch] = min over all possible prev_char of dp[i-1][prev_char]
                        where (prev_char, ch) are not almost-equal
                        plus cost of changing word[i] to ch.
        The final answer is min(dp[n-1][ch]) for ch in ['a'..'z'].
        """
        n = len(word)
        # Convert the original string to a list of numeric codes (0..25).
        arr = [ord(c) - ord('a') for c in word]

        # dp[i][ch] = min cost to fix substring up to i if s[i] = ch
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        # Helper to check if two characters (0..25) are almost-equal
        def almost_equal(x, y):
            return abs(x - y) <= 1
        
        # Initialization for i = 0
        for ch in range(26):
            cost = 0 if ch == arr[0] else 1
            dp[0][ch] = cost
            
        # Fill dp for i > 0
        for i in range(1, n):
            for ch in range(26):
                # cost to change the i-th character to ch
                change_cost = 0 if ch == arr[i] else 1
                # check all possible previous chars
                for prev_ch in range(26):
                    if not almost_equal(prev_ch, ch):
                        dp[i][ch] = min(dp[i][ch], dp[i-1][prev_ch] + change_cost)
                        
        # The answer is the minimum cost possible at index n-1
        return min(dp[n-1])