class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split pattern into three parts using the two '*'
        parts = p.split('*')
        if len(parts) != 3:
            return -1
        
        prefix, middle, suffix = parts
        
        # If empty pattern between stars, return 0
        if not prefix and not middle and not suffix:
            return 0
        
        n = len(s)
        min_length = float('inf')
        
        # For each possible starting position of middle part
        for i in range(n):
            # Check if we can find middle part starting from position i
            if middle:
                mid_pos = s.find(middle, i)
                if mid_pos == -1:
                    break
            else:
                mid_pos = i
                
            # Check prefix before middle part
            if prefix:
                # Find rightmost occurrence of prefix before mid_pos
                prefix_pos = -1
                for j in range(mid_pos - len(prefix) + 1):
                    if s[j:j+len(prefix)] == prefix:
                        prefix_pos = j
                if prefix_pos == -1:
                    continue
            else:
                prefix_pos = 0
                
            # Check suffix after middle part
            if suffix:
                # Find leftmost occurrence of suffix after mid_pos + len(middle)
                suffix_pos = s.find(suffix, mid_pos + len(middle))
                if suffix_pos == -1:
                    continue
                suffix_end = suffix_pos + len(suffix)
            else:
                suffix_end = mid_pos + len(middle)
            
            # Calculate length of matching substring
            if prefix:
                start = prefix_pos
            else:
                start = mid_pos
                
            length = suffix_end - start
            min_length = min(min_length, length)
            
            # If middle is empty, we don't need to try other positions
            if not middle:
                break
            
            i = mid_pos + 1
            
        return min_length if min_length != float('inf') else -1