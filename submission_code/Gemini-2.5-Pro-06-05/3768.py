class Solution:
    def hasSameDigits(self, s: str) -> bool:
        """
        Repeatedly transforms a string of digits until it has a length of two.
        In each transformation, a new string is formed by the sum modulo 10
        of each consecutive pair of digits. The function returns true if the
        final two digits are the same, and false otherwise.
        """
        
        # The loop continues as long as the string has more than two digits.
        # The problem constraints (s.length >= 3) ensure this loop will
        # execute at least once.
        while len(s) > 2:
            # A list comprehension provides a concise and Pythonic way to generate
            # the sequence of new digits for the next iteration.
            # For each pair of adjacent digits s[i] and s[i+1]:
            # 1. Convert them to integers.
            # 2. Calculate their sum modulo 10.
            # 3. Convert the result back to a string.
            # The "".join() method then combines these new digit strings.
            s = "".join([
                str((int(s[i]) + int(s[i+1])) % 10)
                for i in range(len(s) - 1)
            ])
            
        # After the loop, the length of 's' is 2. The result is determined by
        # whether its two digits are identical.
        return s[0] == s[1]