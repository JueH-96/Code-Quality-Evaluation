class Solution:
    def validStrings(self, n: int) -> List[str]:
        from typing import List

        # We will build all valid binary strings of length n by backtracking.
        # A valid string is one where every length-2 substring has at least one '1',
        # which is equivalent to "no two consecutive zeroes."
        
        results = []

        def backtrack(path: str):
            # If we've reached length n, append to results.
            if len(path) == n:
                results.append(path)
                return

            # Option 1: add '0' if it doesn't create "00"
            if not path or path[-1] != '0':
                backtrack(path + '0')

            # Option 2: add '1' always valid
            backtrack(path + '1')

        backtrack("")
        return results