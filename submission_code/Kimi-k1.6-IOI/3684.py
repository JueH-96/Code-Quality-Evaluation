class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_pos = p.index('*')
        left = p[:star_pos]
        right = p[star_pos+1:]
        
        if left and right:
            left_len = len(left)
            right_len = len(right)
            # Iterate over all possible starting positions for the left part
            for i in range(len(s) - left_len + 1):
                if s[i:i+left_len] == left:
                    # Check if the right part exists starting from i + left_len onwards
                    pos = s.find(right, i + left_len)
                    if pos != -1:
                        return True
            return False
        elif not left:  # Only right part exists
            return right in s
        else:  # Only left part exists
            return left in s