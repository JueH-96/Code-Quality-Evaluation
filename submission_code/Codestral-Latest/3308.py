class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Create a set to keep track of characters that have been removed
        removed_chars = set()

        # Iterate over the string in reverse order
        for char in reversed(s):
            # If the character has not been removed yet, add it to the set
            if char not in removed_chars:
                removed_chars.add(char)
            # If all characters from 'a' to 'z' have been removed, return the current substring
            if len(removed_chars) == 26:
                return s[:s.index(char)]

        # If the loop completes, return the original string
        return s