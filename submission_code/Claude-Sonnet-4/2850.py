class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # All AB strings can be used
        result = z * 2
        
        # For AA and BB strings, we need to alternate to avoid AAA or BBB
        # The maximum we can use is limited by the alternating pattern
        if x <= y:
            # Use all x AA's and at most x+1 BB's
            result += x * 2 + min(y, x + 1) * 2
        else:
            # Use all y BB's and at most y+1 AA's  
            result += y * 2 + min(x, y + 1) * 2
            
        return result