import math # No external libraries are needed, math is not used.

class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        Checks if str2 can be made a subsequence of str1 by performing the 
        character increment operation at most once on str1.

        Args:
            str1: The first string, which can be modified by the operation.
            str2: The second string, which we want to check if it's a subsequence 
                  of the potentially modified str1.

        Returns:
            True if str2 can be made a subsequence of str1 with at most one operation,
            False otherwise.
        """
        
        n1 = len(str1)
        n2 = len(str2)

        # Optimization: If str2 is longer than str1, it can never be a subsequence,
        # even after modifications.
        if n2 > n1:
            return False

        p1 = 0  # Pointer for str1
        p2 = 0  # Pointer for str2

        # Helper function to get the next character cyclically ('a' follows 'z').
        def get_next_char(c: str) -> str:
            """Calculates the next character in alphabetical order, wrapping 'z' to 'a'."""
            if c == 'z':
                return 'a'
            else:
                # Use ord() to get ASCII value, increment, and chr() to convert back.
                return chr(ord(c) + 1)

        # Iterate through str1 using pointer p1 and str2 using pointer p2.
        # We try to match characters of str2 in order using characters from str1.
        while p1 < n1 and p2 < n2:
            # Get the current characters from both strings.
            char1 = str1[p1]
            char2 = str2[p2]

            # Check if the current character in str1 (str1[p1]) can match the 
            # current target character in str2 (str2[p2]).
            # A match occurs if str1[p1] is already equal to str2[p2], 
            # OR if str1[p1] becomes equal to str2[p2] after the increment operation.
            if char1 == char2 or get_next_char(char1) == char2:
                # If a match is found (either directly or via increment),
                # we have successfully matched the character str2[p2].
                # We then move to the next character in str2 to find its match.
                p2 += 1

            # Regardless of whether a match was found for str2[p2] at the current str1[p1],
            # we always move to the next character in str1 (increment p1). 
            # This is because we need to find the matches for str2 characters in the correct 
            # relative order within str1, and we consider each character of str1 at most once.
            p1 += 1

        # After iterating through str1 (or until str2 is fully matched), 
        # we check if the pointer p2 has reached the end of str2.
        # If p2 equals the length of str2, it means all characters of str2 
        # were successfully matched in order within str1 (potentially using the 
        # increment operation for some matches).
        return p2 == n2