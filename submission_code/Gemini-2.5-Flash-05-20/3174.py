class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        
        # Iterate through the string taking steps of 2,
        # checking pairs of characters at (i, i+1) where i is even.
        # According to the derived property, for a string to be beautiful,
        # s[i] must be equal to s[i+1] for all even i.
        for i in range(0, n, 2):
            # If the characters in a pair are different, we need to make one change.
            # It doesn't matter if we change s[i] to match s[i+1] or vice-versa,
            # it always costs 1 change for that pair.
            if s[i] != s[i+1]:
                changes += 1
                
        return changes