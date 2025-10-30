class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        """
        We want to ensure that no two adjacent characters in the transformed word
        are 'almost-equal'. Two characters a and b are 'almost-equal' if:

          1) a == b, or
          2) they are adjacent in the English alphabet (e.g., 'a' and 'b').

        We can perform operations where we pick any index i and change word[i] to
        any lowercase English letter. Our goal is to achieve a word with no
        adjacent almost-equal characters using the minimum possible operations.

        Approach:
        Use dynamic programming. Let N be the length of the string, and map
        each character 'a'..'z' to integers 0..25.

        Define DP[i][c]:
          - i: current position in the string (0-based index).
          - c: an integer 0..25 representing letters 'a'+c.

        DP[i][c] = minimum cost (number of changes) to fix positions 0..i in word
                   such that the character at position i is 'a'+c, 
                   and no adjacent pair is almost-equal.

        Transitions:
        DP[i][c] = min over all valid c_prev of [DP[i-1][c_prev] + cost of mapping word[i] to c],
                   subject to 'a'+c_prev is not almost-equal to 'a'+c.

        We then take the minimum of DP[N-1][c] for c in 0..25.

        Time Complexity: O(N * 26 * 26), which is feasible for N up to 100.
        """

        n = len(word)
        if n <= 1:
            # If there's only 0 or 1 character, no adjacent pairs exist => cost is 0
            return 0

        # Convert character to index 0..25
        def char_to_idx(ch: str) -> int:
            return ord(ch) - ord('a')

        # Cost to change from original char to a new char
        def change_cost(orig: str, new_idx: int) -> int:
            return 0 if char_to_idx(orig) == new_idx else 1

        # Check if two characters (by index) are almost-equal
        # They are almost-equal if abs difference <= 1
        def almost_equal_idx(c1: int, c2: int) -> bool:
            return abs(c1 - c2) <= 1

        # Initialize DP array
        # DP[i][c] = large number initially
        INF = 10**9
        DP = [[INF]*26 for _ in range(n)]

        # Base case for i = 0
        for c in range(26):
            DP[0][c] = change_cost(word[0], c)

        # Fill DP
        for i in range(1, n):
            for c in range(26):
                # If we set word[i] to 'a' + c, cost is cost to change
                cost_here = change_cost(word[i], c)
                # We try all possible previous letters c_prev, ensuring
                # that c_prev and c are NOT almost-equal
                best_prev = INF
                for c_prev in range(26):
                    if not almost_equal_idx(c_prev, c):
                        best_prev = min(best_prev, DP[i-1][c_prev])
                DP[i][c] = best_prev + cost_here

        return min(DP[n-1])