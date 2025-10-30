class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*'
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos + 1:]
        
        # Special cases
        if not prefix:
            return suffix in s
        if not suffix:
            return prefix in s
        
        # Find all occurrences of prefix in s and check if suffix can follow
        i = 0
        while True:
            prefix_pos = s.find(prefix, i)
            if prefix_pos == -1:
                break
            prefix_end = prefix_pos + len(prefix)
            
            # Check if suffix appears at or after prefix_end
            if s.find(suffix, prefix_end) != -1:
                return True
            
            i = prefix_pos + 1
        
        return False