class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the count of key changes
        key_changes_count = 0
        
        # Iterate through the string starting from the second character (index 1).
        # We need at least two characters to compare and identify a potential key change.
        # If the string length is less than 2 (i.e., 0 or 1 character), 
        # this loop range will be empty (e.g., range(1,1) or range(1,0)).
        # In such cases, key_changes_count will remain 0, which is correct as no changes are possible.
        # The problem constraint 1 <= s.length <= 100 means s will always have at least one character.
        for i in range(1, len(s)):
            # Get the current character s[i] and the previous character s[i-1].
            # Convert both to lowercase for case-insensitive comparison.
            # This addresses the note that modifiers like shift or caps lock don't count as changing the key.
            # For example, 'a' and 'A' are considered the same key.
            current_char_lower = s[i].lower()
            previous_char_lower = s[i-1].lower()
            
            # If the lowercase versions are different, it means a different key was pressed.
            if current_char_lower != previous_char_lower:
                key_changes_count += 1
                
        return key_changes_count