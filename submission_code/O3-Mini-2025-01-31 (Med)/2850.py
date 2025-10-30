class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We'll use recursion with memoization.
        # We'll denote our pieces as:
        #   Type 'A': "AA"   (count = x)
        #   Type 'B': "BB"   (count = y)
        #   Type 'C': "AB"   (count = z)
        #
        # Their first letters:
        #   A: 'A'
        #   B: 'B'
        #   C: 'A'
        # Their last letters:
        #   A: 'A'
        #   B: 'B'
        #   C: 'B'
        #
        # Since pieces "AA" and "BB" are dangerous when placed consecutively,
        # we analyze safe transitions by considering the last piece used.
        # Rules: We must not create triple consecutive 'A' or 'B'.
        #
        # Notice that each piece is of length 2 and within pieces A and B,
        # the pair letters are identical (hence a run of 2).
        # The AB piece (C) doesn't produce a run of 2 of the same letter.
        #
        # When concatenating, say we have a previous piece ending with letter L 
        # and its last piece produces a run of 2 if it is A or B.
        # So if the previous piece is either A ("AA") or B ("BB") or C ("AB") that ends with B,
        # then to avoid triple we must ensure that if the previous piece produced a run of 2,
        # the next piece's first letter is different from that letter.
        #
        # From analyzing each piece type:
        #   If the previous piece was A ("AA"), it ended in A and in fact provided two consecutive A's.
        #       So the next piece MUST NOT start with A. 
        #       Which piece(s) start with B? Only piece B ("BB").
        #   If the previous piece was B ("BB"), it ended in B (a run of 2).
        #       So the next piece MUST NOT start with B.
        #       Which pieces start with A? Pieces A ("AA") and C ("AB").
        #   If the previous piece was C ("AB"), it ends in B,
        #       so again the next piece MUST NOT start with B.
        #       So allowed: pieces starting with A => A ("AA") and C ("AB").
        #
        # For the very first piece, there is no previous constraint.
        #
        # Now we perform a DFS with state (a, b, c, last) where
        #   a = number of remaining "AA" pieces,
        #   b = number of remaining "BB" pieces,
        #   c = number of remaining "AB" pieces,
        #   last = the type of the last piece used (None, 'A', 'B', or 'C').
        #
        # Our dp returns the maximum number of pieces (each of length 2) that we can use,
        # and finally we multiply by 2.
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(a, b, c, last):
            best = 0
            # Based on the last piece used, determine allowed moves:
            if last is None:
                # Can choose any piece that is available.
                if a > 0:
                    best = max(best, 1 + dp(a-1, b, c, 'A'))
                if b > 0:
                    best = max(best, 1 + dp(a, b-1, c, 'B'))
                if c > 0:
                    best = max(best, 1 + dp(a, b, c-1, 'C'))
            elif last == 'A':  # last piece was "AA" (ends with A, run=2) 
                # Next piece MUST not start with A. Only allowed is a piece that starts with B.
                # Only type B ("BB") starts with B.
                if b > 0:
                    best = max(best, 1 + dp(a, b-1, c, 'B'))
            elif last == 'B':  # last piece was "BB" (ends with B, run=2)
                # Next piece MUST not start with B. Allowed are pieces starting with A: A and C.
                if a > 0:
                    best = max(best, 1 + dp(a-1, b, c, 'A'))
                if c > 0:
                    best = max(best, 1 + dp(a, b, c-1, 'C'))
            elif last == 'C':  # last piece was "AB" (ends with B but did not produce a run of B's because it is only a single B at the end)
                # However, notice that the danger arises from having two repeated letters.
                # Since "AB" only gives a single B at the end, one might wonder if a piece starting with B is allowed.
                # But careful: The rule of the game is given by the fact that the pieces "BB" always produce a run of 2 B's.
                # Since last piece "AB" ends with B and does not have a doubling effect, if we choose a B type,
                # the new piece "BB" would yield a run B, B, so that's fine as long as it does not create three.
                # However, that is a bit subtle because in our setup, we note that a violation happens only when a piece that with built-in run 2
                # is appended right after a piece ending with the same letter.
                # We use the safe transitions from our initial analysis: after a "C" (ends with B), we cannot choose a piece that starts with B because:
                #   "C" always produces "AB", so the last letter is B and it does not form a pair, but after "C", if you use a piece starting with B,
                #   you will have just a single B concatenated with a piece that starts with B (which gives a built-in double B). 
                #   Then the result will be .. B + (B and B) i.e. three consecutive B's? Let’s check: "AB" + "BB" = A, B, B, B.
                #   Thus, after C, allowed pieces are those starting with A: A ("AA") and C ("AB").
                if a > 0:
                    best = max(best, 1 + dp(a-1, b, c, 'A'))
                if c > 0:
                    best = max(best, 1 + dp(a, b, c-1, 'C'))
            return best
        
        # dp returns maximum count of pieces, each piece adds 2 to the length.
        max_pieces = dp(x, y, z, None)
        return max_pieces * 2


# For debugging or testing purposes:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: Input: x = 2, y = 5, z = 1 --> Expected output: 12
    print(sol.longestString(2, 5, 1))  # Expected 12

    # Example 2: Input: x = 3, y = 2, z = 2 --> Expected output: 14
    print(sol.longestString(3, 2, 2))  # Expected 14