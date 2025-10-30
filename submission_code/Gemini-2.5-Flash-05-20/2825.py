class Solution:
    def minimizedStringLength(self, s: str) -> int:
        """
        Given a 0-indexed string s, repeatedly perform an operation to minimize its length.
        The operation involves choosing an index i, and deleting the closest occurrence of
        s[i] to its left and right.

        The key insights are:
        1. The character s[i] itself is never deleted. This means if a character 'c' is
           present in the original string, at least one instance of 'c' will always remain.
        2. Operations on a specific character type (e.g., 'a') only affect other characters
           of the same type. They do not affect 'b's, 'c's, etc. This makes the minimization
           process for each unique character independent.

        Combining these two points:
        - For every unique character 'char_val' present in the original string 's', we can
          always perform operations to reduce its multiple occurrences down to just one.
          (e.g., if you have "dddd", picking the second 'd' removes the first and third,
          leaving "dd". Then picking the first 'd' removes the second, leaving "d".)
        - Since we cannot remove the last instance of any character type, and operations
          are independent across character types, the minimized string will contain exactly
          one instance of each unique character that was initially present.

        Therefore, the minimum length of the string will be equal to the number of unique
        characters in the original string.
        """
        
        # Convert the string to a set to automatically get all unique characters.
        unique_chars = set(s)
        
        # The length of this set is the number of unique characters,
        # which is the minimized length of the string.
        return len(unique_chars)