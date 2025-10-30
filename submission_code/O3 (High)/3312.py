class Solution:
    def countKeyChanges(self, s: str) -> int:
        """
        Counts how many times the user had to change the key while typing the
        string `s`. A change of key occurs when two consecutive characters,
        after ignoring case (since 'a' and 'A' are the same physical key),
        differ.
        
        :param s: The typed string
        :return: Number of key changes
        """
        if len(s) <= 1:
            return 0
        
        changes = 0
        for i in range(1, len(s)):
            # Lowercase both characters to compare their physical keys
            if s[i].lower() != s[i - 1].lower():
                changes += 1
                
        return changes