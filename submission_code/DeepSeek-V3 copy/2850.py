class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The key is to balance the counts of 'AA' and 'BB' to avoid 'AAA' or 'BBB'
        # Each 'AA' must be followed by a 'BB' and vice versa, except for the last one
        # 'AB' can be placed anywhere without affecting the balance
        
        # The maximum number of 'AA' and 'BB' we can use is min(2 * x, 2 * y) + max(x, y) - min(x, y)
        # But a simpler approach is to use min(x, y) pairs of 'AA' and 'BB', and then add the remaining 'AA' or 'BB' if any
        
        # Calculate the number of pairs of 'AA' and 'BB'
        pairs = min(x, y)
        # The remaining 'AA' or 'BB' can be added, but only one more
        remaining = max(x, y) - pairs
        # Each 'AB' contributes 2 characters and can be added without restriction
        # So the total length is pairs * 4 (since each pair is 'AA' + 'BB') + remaining * 2 + z * 2
        total = pairs * 4 + remaining * 2 + z * 2
        return total