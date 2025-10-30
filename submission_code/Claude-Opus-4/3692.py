class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Find positions of the two asterisks
        first_star = p.index('*')
        second_star = p.rindex('*')
        
        # Split pattern into three parts
        prefix = p[:first_star]
        middle = p[first_star + 1:second_star]
        suffix = p[second_star + 1:]
        
        min_length = float('inf')
        
        # Try each possible starting position in s
        for start in range(len(s) + 1):
            # Check if we can match prefix starting at position 'start'
            if start + len(prefix) > len(s):
                continue
            if prefix and s[start:start + len(prefix)] != prefix:
                continue
            
            # Start searching for middle part after prefix
            middle_start = start + len(prefix)
            
            # Try each possible position for middle part
            for mid_pos in range(middle_start, len(s) + 1):
                # Check if we can match middle starting at position 'mid_pos'
                if mid_pos + len(middle) > len(s):
                    continue
                if middle and s[mid_pos:mid_pos + len(middle)] != middle:
                    continue
                
                # Start searching for suffix after middle
                suffix_start = mid_pos + len(middle)
                
                # Try each possible position for suffix
                for suf_pos in range(suffix_start, len(s) + 1):
                    # Check if we can match suffix starting at position 'suf_pos'
                    if suf_pos + len(suffix) > len(s):
                        continue
                    if suffix and s[suf_pos:suf_pos + len(suffix)] != suffix:
                        continue
                    
                    # Calculate the length of this match
                    end = suf_pos + len(suffix)
                    length = end - start
                    min_length = min(min_length, length)
        
        return min_length if min_length != float('inf') else -1