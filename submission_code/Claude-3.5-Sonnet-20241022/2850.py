class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # For maximum length, we want to avoid AAA and BBB
        # We can use all AB strings as they don't create problematic sequences
        # For AA and BB strings, we want to alternate them to avoid AAA and BBB
        
        # Use all AB strings
        result = 2 * z
        
        # For AA and BB strings:
        # If x == y, we can use all AA and BB strings alternating
        # If x > y, we can use y pairs of AA/BB plus one more AA
        # If y > x, we can use x pairs of AA/BB plus one more BB
        
        if x == y:
            result += 4 * x
        else:
            result += 4 * min(x, y) + 2
            
        return result