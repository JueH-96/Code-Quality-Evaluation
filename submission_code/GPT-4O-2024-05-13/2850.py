class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We need to avoid "AAA" and "BBB" substrings.
        # Let's try to balance the number of "AA" and "BB" strings as much as possible.
        
        # The maximum number of "AA" and "BB" strings we can use without causing "AAA" or "BBB"
        max_pairs = min(x, y)
        
        # We can use all "AB" strings since they don't contribute to "AAA" or "BBB"
        total_length = 2 * (x + y + z)
        
        # We need to subtract the length of the unused "AA" or "BB" strings if they are more than the pairs we can use
        if x > y:
            total_length -= 2 * (x - max_pairs)
        elif y > x:
            total_length -= 2 * (y - max_pairs)
        
        return total_length