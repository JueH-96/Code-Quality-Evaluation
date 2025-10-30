class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(x, y, z, last):
            # Base case: no more strings to place
            if x == 0 and y == 0 and z == 0:
                return 0
            
            max_len = 0
            
            # Try placing "AA" (encoded as 1)
            if x > 0 and last in {0, 2, 3}:  # Can place AA after initial, BB, or AB
                max_len = max(max_len, 2 + dp(x-1, y, z, 1))
            
            # Try placing "BB" (encoded as 2)  
            if y > 0 and last in {0, 1}:  # Can place BB after initial or AA
                max_len = max(max_len, 2 + dp(x, y-1, z, 2))
            
            # Try placing "AB" (encoded as 3)
            if z > 0 and last in {0, 2, 3}:  # Can place AB after initial, BB, or AB
                max_len = max(max_len, 2 + dp(x, y, z-1, 3))
            
            return max_len
        
        # Start with no previous string (encoded as 0)
        return dp(x, y, z, 0)