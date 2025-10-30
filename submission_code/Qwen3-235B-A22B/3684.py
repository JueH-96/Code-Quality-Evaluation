class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of the star in the pattern
        star_pos = p.index('*')
        left = p[:star_pos]
        right = p[star_pos + 1:]
        l = len(left)
        r = len(right)
        
        # Iterate over all possible starting positions for the left part in s
        for i in range(len(s) - l + 1):
            # Check if the substring matches the left part
            if s[i:i + l] != left:
                continue
            
            # Calculate the minimum and maximum possible positions for the right part
            k_min = i + l
            k_max = len(s) - r
            
            # If there's no valid range for k, skip
            if k_min > k_max:
                continue
            
            # Special case when right is empty (r == 0)
            if r == 0:
                return True
            
            # Check each possible k in the valid range
            for k in range(k_min, k_max + 1):
                if s[k:k + r] == right:
                    return True
        
        # No valid match found
        return False