class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We have three kinds of blocks:
        #   "AA"  (x available) – starts with A, ends with A.
        #   "BB"  (y available) – starts with B, ends with B.
        #   "AB"  (z available) – starts with A, ends with B.
        #
        # When concatenating blocks, we must avoid having three consecutive A’s or B’s.
        # In any adjacent pair the “junction” is formed by the last character of the previous
        # block and the first character of the next block. Notice that:
        #   • A block that is “AA” ends with A; if the next block begins with A that would yield AAA.
        #   • A block that is “BB” ends with B; if the next block begins with B that would yield BBB.
        #
        # Thus when choosing the next block our decision is forced by the last block’s ending letter:
        #   - If the previous block ended with A then the next block must start with B.
        #     Among our options, only the "BB" block (which starts with B) is allowed.
        #   - If the previous block ended with B then the next block must start with A.
        #     Our blocks that start with A are "AA" and "AB".
        #   - When no block has been chosen yet (start of string) there’s no restriction.
        #
        # Note that if you use an "AA" block while in a state where A is not allowed to be repeated,
        # you must “recover” by following it with a "BB" block (if you wish to continue) because:
        #   • In state B, you may use "AA" as a move – which sends the sequence into a state ending with A.
        #     But from state A the only allowed next block is "BB" (to avoid AAA).
        #   • On the other hand, note that an "AB" block (when allowed) always sends you to state B.
        #
        # So a natural recurrence is to define a function dp(last, a, b, c)
        # where last is the ending letter of the current concatenated string:
        #   • last can be "A", "B", or "None" (meaning no block chosen yet).
        #   • a, b, c are the remaining counts of blocks "AA", "BB", "AB" respectively.
        #
        # The recurrence is:
        #   If last is None, you may choose any available block:
        #     - if you use "AA": new state becomes A.
        #     - if you use "BB": new state becomes B.
        #     - if you use "AB": new state becomes B.
        #
        #   If last == "A", then the previous block ended with A.
        #     To avoid AAA the next block must start with B – that forces the block "BB".
        #
        #   If last == "B", then the next block must start with A.
        #     You may choose either "AA" (which changes the state to A) or "AB" (keeps state B).
        #
        # Our goal is to maximize the number of blocks selected. Since every block is length 2,
        # maximizing the block count yields the maximum possible length (which is 2 * number_of_blocks).
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(last: str, a: int, b: int, c: int) -> int:
            best = 0
            # When no block has been placed yet.
            if last == "None":
                if a > 0:
                    best = max(best, 1 + dp("A", a - 1, b, c))
                if b > 0:
                    best = max(best, 1 + dp("B", a, b - 1, c))
                if c > 0:
                    best = max(best, 1 + dp("B", a, b, c - 1))
            elif last == "A":
                # Previous block ended with A.
                # To avoid "AAA", the next block must start with B.
                # Only block starting with B is "BB".
                if b > 0:
                    best = max(best, 1 + dp("B", a, b - 1, c))
            elif last == "B":
                # Last block ended with B.
                # Next block must start with A, so we can use either "AA" or "AB".
                if a > 0:
                    best = max(best, 1 + dp("A", a - 1, b, c))
                if c > 0:
                    best = max(best, 1 + dp("B", a, b, c - 1))
            return best
        
        # We use "None" (a string) as the initial state indicator.
        max_blocks = dp("None", x, y, z)
        return max_blocks * 2  # each block is of length 2

# Below is some test code:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: x = 2, y = 5, z = 1  --> Output should be 12
    print(sol.longestString(2, 5, 1))
    # Example 2: x = 3, y = 2, z = 2  --> Output should be 14
    print(sol.longestString(3, 2, 2))