class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The maximum number of "AA" and "BB" pairs we can use is the minimum of x and y
        min_pairs = min(x, y)
        
        # We can use all "AB" strings
        total_length = z * 2
        
        # Add the "AA" and "BB" pairs
        total_length += min_pairs * 4
        
        # If we have extra "AA" or "BB" strings, we can use one more pair
        if x != y:
            total_length += 2
        
        return total_length