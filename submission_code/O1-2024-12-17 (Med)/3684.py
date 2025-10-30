class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split p into two parts (left and right) around the '*'
        star_index = p.index('*')
        left = p[:star_index]
        right = p[star_index+1:]
        
        len_left, len_right = len(left), len(right)
        
        # We only need substrings in s of length >= len_left + len_right,
        # because the middle can be any string (including empty) for '*'.
        for start in range(len(s)):
            for end in range(start, len(s)):
                sub = s[start:end+1]
                if len(sub) >= len_left + len_right:
                    # Check if sub starts with left-part and ends with right-part
                    if sub.startswith(left) and sub.endswith(right):
                        return True
        
        return False