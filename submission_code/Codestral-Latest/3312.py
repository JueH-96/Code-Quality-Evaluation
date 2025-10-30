class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the count of key changes
        key_changes = 0

        # Iterate through the string starting from the second character
        for i in range(1, len(s)):
            # Check if the current character is different from the previous one
            # We use lower() to ignore case differences
            if s[i].lower() != s[i-1].lower():
                key_changes += 1

        return key_changes