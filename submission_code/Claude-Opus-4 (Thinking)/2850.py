class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache
        
        # States: 
        # 0 = empty
        # 1 = ends with single 'A' (not AA)
        # 2 = ends with single 'B' (not BB)
        # 3 = ends with "AA"
        # 4 = ends with "BB"
        
        @lru_cache(None)
        def dp(rx, ry, rz, state):
            if rx == 0 and ry == 0 and rz == 0:
                return 0
            
            max_len = 0
            
            # Try to append "AA"
            if rx > 0 and state != 1 and state != 3:  # Can't create "AAA"
                max_len = max(max_len, 2 + dp(rx - 1, ry, rz, 3))
            
            # Try to append "BB"
            if ry > 0 and state != 2 and state != 4:  # Can't create "BBB"
                max_len = max(max_len, 2 + dp(rx, ry - 1, rz, 4))
            
            # Try to append "AB"
            if rz > 0 and state != 3:  # Can't create "AAAB"
                max_len = max(max_len, 2 + dp(rx, ry, rz - 1, 2))
            
            return max_len
        
        return dp(x, y, z, 0)