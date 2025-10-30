from collections import Counter

class Solution:
    def findValidPair(self, s: str) -> str:
        """
        Finds the first valid adjacent pair of digits in a string s.

        A valid pair (d1, d2) must satisfy:
        1. d1 != d2 (the digits are different).
        2. d1 appears in s exactly int(d1) times.
        3. d2 appears in s exactly int(d2) times.

        Args:
            s: A string consisting only of digits '1' through '9'.

        Returns:
            The first valid pair as a two-character string, or an empty string
            if no valid pair exists.
        """
        n = len(s)
        if n < 2:
            return ""

        # Pre-calculate the counts of each digit in the string
        counts = Counter(s)

        # Iterate through adjacent pairs (s[i], s[i+1])
        for i in range(n - 1):
            digit1_char = s[i]
            digit2_char = s[i+1]

            # Condition 1: Digits must be different
            if digit1_char == digit2_char:
                continue # This pair is not valid, move to the next

            # Get the numeric values of the digits
            digit1_val = int(digit1_char)
            digit2_val = int(digit2_char)

            # Condition 2: Check if the count of digit1 matches its value
            # Use counts.get(char, 0) in case a digit from the pair surprisingly
            # doesn't exist in the string (though problem constraints suggest otherwise)
            if counts.get(digit1_char, 0) != digit1_val:
                continue # This pair is not valid, move to the next

            # Condition 3: Check if the count of digit2 matches its value
            if counts.get(digit2_char, 0) != digit2_val:
                continue # This pair is not valid, move to the next

            # If all conditions are met, we've found the first valid pair
            return digit1_char + digit2_char

        # If the loop completes without finding any valid pair
        return ""