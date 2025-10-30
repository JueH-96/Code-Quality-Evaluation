class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The maximum length of the string can be calculated by considering:
        # - We can use all 'AB' strings since they don't contribute to "AAA" or "BBB".
        # - We can use all 'AA' and 'BB' strings, but we need to ensure we don't form "AAA" or "BBB".
        # - The strategy is to alternate between 'AA' and 'BB' as much as possible.
        
        # We can use all 'AB' strings
        length = 2 * z
        
        # We can use min(x, y) pairs of 'AA' and 'BB' without forming "AAA" or "BBB"
        pairs = min(x, y)
        length += 4 * pairs  # Each pair contributes 4 to the length ('AA' + 'BB')
        
        # If there are more 'AA' or 'BB' left, we can add one more of them
        if x > pairs:
            length += 2  # Add one more 'AA'
        elif y > pairs:
            length += 2  # Add one more 'BB'
        
        return length