class Solution:
    def countKeyChanges(self, s: str) -> int:
        """
        Counts the number of times the key changes in a string,
        ignoring case differences.

        Args:
            s: The input string typed by the user.

        Returns:
            The number of key changes.
        """
        count = 0
        # We need at least two characters to have a change.
        # Iterate from the second character (index 1)
        for i in range(1, len(s)):
            # Compare the current character with the previous one,
            # ignoring case by converting both to lowercase.
            if s[i].lower() != s[i-1].lower():
                count += 1
        return count