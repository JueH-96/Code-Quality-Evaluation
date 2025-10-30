class Solution:
    def countKeyChanges(self, s: str) -> int:
        # Initialize the counter for key changes.
        # There are no changes if the string has 0 or 1 character.
        changes = 0
        
        # We start comparing from the second character (index 1)
        # because the first character doesn't have a preceding one to compare against.
        for i in range(1, len(s)):
            # Convert both the current character and the previous character to lowercase.
            # This allows us to compare them case-insensitively, fulfilling the
            # condition that modifiers (like Shift/Caps Lock) don't count as a key change.
            if s[i].lower() != s[i-1].lower():
                # If the lowercase versions are different, it means a different
                # base key was pressed, so we increment the changes counter.
                changes += 1
                
        # Return the total number of key changes.
        return changes