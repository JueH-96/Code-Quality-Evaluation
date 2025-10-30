class Solution:
    def countKeyChanges(self, s: str) -> int:
        """
        Calculates the number of times the key is changed when typing the string s.
        A key change is defined as using a key different from the previous one,
        ignoring case.
        """
        changes = 0
        
        # Iterate from the second character to the end of the string.
        # The loop starts at index 1 to allow comparison with the previous character at index i-1.
        for i in range(1, len(s)):
            # A key change occurs if the current character is different from the
            # previous one. The comparison is made case-insensitive by converting
            # both characters to lowercase.
            if s[i].lower() != s[i-1].lower():
                changes += 1
        
        return changes