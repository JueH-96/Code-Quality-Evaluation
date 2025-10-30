from collections import Counter

class Solution:
    def minimumLength(self, s: str) -> int:
        """
        For every character c that appears k times in the string:
          • one operation always deletes exactly two occurrences of c
          • the pivot character chosen for the operation is never deleted
        Therefore the parity (even / odd) of the number of occurrences of every
        character is preserved throughout all operations.

        If k is odd we can keep deleting pairs until only one c is left.
        If k is even we can keep deleting pairs until two c's are left
        (we can never reach an odd count starting from an even one).

        The minimal length of the final string is consequently
            Σ ( 1  if count(c) is odd
              else 2 )
        taken over all characters that occur in the original string.
        """
        freq = Counter(s)
        return sum(1 if cnt & 1 else 2 for cnt in freq.values())