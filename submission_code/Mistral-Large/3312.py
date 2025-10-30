class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0

        # Convert the string to lower case to ignore case differences
        s = s.lower()

        # Initialize the count of key changes
        key_changes = 0

        # Iterate through the string and count the key changes
        for i in range(1, len(s)):
            if s[i] != s[i-1]:
                key_changes += 1

        return key_changes