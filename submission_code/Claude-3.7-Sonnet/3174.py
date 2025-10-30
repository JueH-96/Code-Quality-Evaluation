class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        
        # Check each pair of characters
        for i in range(0, n, 2):
            # If characters in a pair are different, we need to change one of them
            if s[i] != s[i+1]:
                changes += 1
                
        return changes