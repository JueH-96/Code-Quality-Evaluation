class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We have three types of pieces:
        #   "AA"  with count = x
        #   "BB"  with count = y
        #   "AB"  with count = z
        #
        # We can choose an order (any permutation of chosen pieces) such that when concatenated,
        # the resulting string does not contain "AAA" nor "BBB".
        #
        # Because each piece is fixed and must appear as a whole, we simulate the process of
        # concatenation piece by piece. We keep track of the current trailing letter and how many
        # times that letter has consecutively appeared from the end of the already formed string.
        # This matters because appending a piece might extend that run.
        #
        # For piece "AA":
        #   If the current state's last letter is 'A', then adding "AA" would add 2 more A's.
        #   We must have current run + 2 <= 2 (since we cannot have 3 consecutive same letters).
        #   If the last letter is not 'A' (or if there is no last letter), then appending "AA" will
        #   create a run of 2 A's.
        #
        # For piece "BB": similar rules as above.
        #
        # For piece "AB": This piece has two letters.
        #   We process letter-by-letter.
        #   Let s be the current state with (last, run). Then:
        #     - Append 'A':
        #         if last == 'A', new_run = run + 1; else new_run = 1.
        #         We must have new_run <= 2.
        #     - Then append 'B':
        #         Since 'B' != 'A', the run resets to 1 and last becomes 'B'.
        #
        # We use DFS with memoization on the state defined by (x_count, y_count, z_count, last, run)
        # where last is the last character in our concatenated string (or None if empty) and run is
        # the number of consecutive occurrences for that last character.
        #
        # The function returns the maximum additional length that can be achieved from that state.
        
        from functools import lru_cache
        
        # Define a helper that attempts to append a piece to the current state.
        # It returns the new state (last, run) and the length added by that piece (which is fixed)
        # if the move is valid, or None if it violates the rule.
        def apply_piece(piece, last, run):
            # piece will be one of: "AA", "BB", "AB"
            # We simulate letter by letter.
            if piece == "AA":
                # Check letter by letter, but note piece letters are same.
                if last == 'A':
                    # Then we add two more A's: run + 2.
                    if run + 2 > 2:  # would yield 3 or more
                        return None
                    else:
                        return ('A', run + 2)
                else:
                    # If last is None or last != 'A', then we start a new run of A's, length 2.
                    return ('A', 2)
            elif piece == "BB":
                if last == 'B':
                    if run + 2 > 2:
                        return None
                    else:
                        return ('B', run + 2)
                else:
                    return ('B', 2)
            elif piece == "AB":
                # Process first letter 'A', then letter 'B'.
                # Step 1: append 'A'
                if last == 'A':
                    new_run = run + 1
                else:
                    new_run = 1
                if new_run > 2:
                    return None
                # Now, after adding 'A', the temporary state is ('A', new_run)
                # Step 2: Append 'B'
                # Since 'B' is different from 'A', we simply reset run = 1.
                return ('B', 1)
            else:
                return None

        @lru_cache(maxsize=None)
        def dfs(a_count, b_count, ab_count, last, run):
            best = 0
            # Try using a piece among the available counts.
            # "AA"
            if a_count > 0:
                new_state = apply_piece("AA", last, run)
                if new_state is not None:
                    new_last, new_run = new_state
                    res = 2 + dfs(a_count - 1, b_count, ab_count, new_last, new_run)
                    if res > best:
                        best = res
            # "BB"
            if b_count > 0:
                new_state = apply_piece("BB", last, run)
                if new_state is not None:
                    new_last, new_run = new_state
                    res = 2 + dfs(a_count, b_count - 1, ab_count, new_last, new_run)
                    if res > best:
                        best = res
            # "AB"
            if ab_count > 0:
                new_state = apply_piece("AB", last, run)
                if new_state is not None:
                    new_last, new_run = new_state
                    res = 2 + dfs(a_count, b_count, ab_count - 1, new_last, new_run)
                    if res > best:
                        best = res
            return best

        # initial state: no last character, run=0.
        # Use None as a special marker for last character.
        return dfs(x, y, z, None, 0)


# Below are some tests you can run to validate your solution.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.longestString(2, 5, 1))  # Expected 12
    # Example 2:
    print(sol.longestString(3, 2, 2))  # Expected 14
    # Additional tests:
    print(sol.longestString(1, 1, 1))  # some small input
    print(sol.longestString(50, 50, 50))  # maximum limits test