class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split pattern by '*'
        parts = p.split('*')
        prefix, middle, suffix = parts[0], parts[1], parts[2]
        
        min_length = float('inf')
        n = len(s)
        
        # For each starting position
        for i in range(n + 1):
            # Check if prefix matches at position i
            if i + len(prefix) > n:
                continue
            if len(prefix) > 0 and s[i:i + len(prefix)] != prefix:
                continue
            
            # For each ending position
            for j in range(i + len(prefix), n + 1):
                # Check if suffix matches before position j
                if j < len(suffix):
                    continue
                if len(suffix) > 0 and s[j - len(suffix):j] != suffix:
                    continue
                
                # Check if middle exists in between
                middle_part = s[i + len(prefix):j - len(suffix)]
                if len(middle) == 0 or middle in middle_part:
                    min_length = min(min_length, j - i)
        
        return min_length if min_length != float('inf') else -1