from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        """
        Return all binary strings of length n that do not contain '00' as a substring.
        Equivalent to all strings where every pair of consecutive bits has at least one '1'.
        """
        res = []
        
        def backtrack(prefix: str, last_was_zero: bool):
            # If we've built a string of length n, record it.
            if len(prefix) == n:
                res.append(prefix)
                return
            # We can always append '1'.
            backtrack(prefix + '1', False)
            # We append '0' only if the last character wasn't '0' (to avoid '00').
            if not last_was_zero:
                backtrack(prefix + '0', True)
        
        # Start with an empty prefix; treat as if last was not zero.
        backtrack("", False)
        return res