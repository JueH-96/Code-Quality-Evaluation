class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Each AB contributes 2 characters and we can use all of them
        result = z * 2
        
        # For AA and BB, we need to alternate to avoid AAA or BBB
        # If we have equal amounts, we can use all of both
        if x == y:
            result += x * 2 + y * 2
        # If we have more AA than BB
        elif x > y:
            # We can use all BB and one extra AA
            result += y * 2 + (y + 1) * 2
        # If we have more BB than AA
        else:
            # We can use all AA and one extra BB
            result += x * 2 + (x + 1) * 2
            
        return result