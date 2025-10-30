class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the count of key changes
        key_changes = 0
        
        # Iterate over the string from the second character
        for i in range(1, len(s)):
            # Compare the current character with the previous one
            # Convert both to lowercase to ignore case differences
            if s[i].lower() != s[i-1].lower():
                key_changes += 1
        
        return key_changes