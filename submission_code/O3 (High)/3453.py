from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        """
        Generate all binary strings of length `n` that do **not**
        contain the substring "00" (equivalently: every substring
        of length 2 has at least one '1').
        """
        res: List[str] = []

        def backtrack(idx: int, prev_zero: bool, cur: List[str]) -> None:
            """
            idx        -> current position we are about to fill (0-based)
            prev_zero  -> True  if the previous character placed was '0'
            cur        -> list with the characters chosen so far
            """
            if idx == n:                      # full string built
                res.append(''.join(cur))
                return

            # Try placing '0' first (lexicographically smaller),
            # allowed only if previous char is not '0'.
            if not prev_zero:
                cur.append('0')
                backtrack(idx + 1, True, cur)
                cur.pop()

            # Always allowed to place '1'.
            cur.append('1')
            backtrack(idx + 1, False, cur)
            cur.pop()

        backtrack(0, False, [])
        return res