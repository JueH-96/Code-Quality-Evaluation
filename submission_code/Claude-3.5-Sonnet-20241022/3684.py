class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split pattern into prefix and suffix at '*'
        prefix, suffix = p.split('*')
        
        # If prefix + suffix is longer than s, it can't be a substring
        if len(prefix) + len(suffix) > len(s):
            return False
            
        # Check if s has the prefix and suffix at valid positions
        for i in range(len(s) - len(suffix)):
            # Check if substring starting at i could be valid
            if len(s) - i < len(prefix):
                break
                
            # Check if prefix matches at current position
            if s[i:i+len(prefix)] == prefix:
                # If no suffix, we found a match
                if not suffix:
                    return True
                    
                # Check all possible positions for suffix after prefix
                start_pos = i + len(prefix)
                for j in range(start_pos, len(s) - len(suffix) + 1):
                    if s[j:j+len(suffix)] == suffix:
                        return True
                        
        return False