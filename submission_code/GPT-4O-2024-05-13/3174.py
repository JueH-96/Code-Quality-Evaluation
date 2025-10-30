class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        
        # Iterate through the string in steps of 2
        for i in range(0, n, 2):
            # Check the pair of characters
            if s[i] != s[i + 1]:
                # If they are different, we need to change one of them
                changes += 1
        
        return changes