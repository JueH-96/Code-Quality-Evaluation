class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the count of key changes
        key_changes = 0
        
        # If the string is empty or has only one character, there are no key changes
        if len(s) < 2:
            return key_changes
        
        # Iterate over the string from the second character to the end
        for i in range(1, len(s)):
            # Check if the current character is different from the previous one (ignoring case)
            if s[i].lower() != s[i-1].lower():
                # If they are different, increment the count of key changes
                key_changes += 1
        
        # Return the total count of key changes
        return key_changes