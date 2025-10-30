class Solution:
    def hasSameDigits(self, s: str) -> bool:
        """
        Repeatedly replaces the current digit list with the list of pairwise sums
        taken modulo 10 until only two digits remain. Returns True if those two
        digits are identical, otherwise False.
        """
        digits = [int(ch) for ch in s]             # Convert to list of integers

        while len(digits) > 2:                     # Continue until only two left
            digits = [(digits[i] + digits[i + 1])  # New digit = (a+b) mod 10
                      % 10
                      for i in range(len(digits) - 1)]

        return digits[0] == digits[1]              # Compare the final two digits