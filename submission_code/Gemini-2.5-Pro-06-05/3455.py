import collections

class Solution:
    def minimumLength(self, s: str) -> int:
        # Count the frequency of each character in the string.
        counts = collections.Counter(s)
        
        # The core idea is that operations on a character type are independent
        # of other character types.
        # For any character `c` with count `k`:
        # - An operation requires a 'center' `c`, a `c` to its left, and a `c` to its right.
        #   This means at least 3 `c`s are needed for an operation.
        # - Each operation removes 2 `c`s.
        # - This process continues until the number of `c`s is less than 3.
        # - If `k` is odd, the count can be reduced as 5 -> 3 -> 1. One character remains.
        # - If `k` is even, the count can be reduced as 6 -> 4 -> 2. Two characters remain.
        
        # We sum the number of remaining characters for each type.
        # A generator expression with sum() is a concise way to do this.
        return sum(1 if k % 2 == 1 else 2 for k in counts.values())