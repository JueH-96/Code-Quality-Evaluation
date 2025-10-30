class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        res = float('inf')
        
        # Try all possible parity of the first character
        for first in ['0', '1']:
            changes = 0
            curr = first
            
            # Iterate over the string
            for i in range(n):
                # If the current character does not match the expected character, increment changes
                if s[i] != curr:
                    changes += 1
                
                # Alternate the expected character for the next pair
                curr = '1' if curr == '0' else '0'
            
            # Update the result
            res = min(res, changes)
        
        return res