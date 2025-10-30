class Solution:
    def minChanges(self, s: str) -> int:
        # Initialize the number of changes needed
        changes = 0
        
        # Iterate over the string in steps of 2
        for i in range(0, len(s), 2):
            # Check the pair of characters
            if s[i] != s[i + 1]:
                # If they are different, we need to change one of them
                changes += 1
        
        return changes