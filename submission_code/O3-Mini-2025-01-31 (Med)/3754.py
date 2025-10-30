class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # We will iterate over each prefix.
        # For a prefix of length p, let:
        #  countE, countW, countN, countS be the number of each direction.
        #
        # For horizontal moves:
        #   H = countE + countW, and choosing the best sign (E if countE>=countW,
        #   or W otherwise) gives a baseline horizontal contribution:
        #       base_x = |countE - countW|
        #   and the number of moves “against” the best sign is:
        #       r_h = min(countE, countW)
        #
        # Analogously for vertical moves.
        #
        # Without modifications the Manhattan distance is:
        #   base = base_x + base_y
        #
        # If we “fix” one move (i.e. change the letter) then a move that was misaligned
        # (contributing –1 in its axis) can be made to contribute +1, an improvement of 2.
        #
        # Given that we can modify at most mod = min(k, p) moves over p moves,
        # the best we can achieve is:
        #   candidate = base + 2 * min(mod, r_h + r_v)
        # but we cannot beat p (since every move contributes at most 1 if they all point
        # in the same “good” direction). 
        #
        # We then choose the maximum candidate over all prefixes.
        
        max_val = 0
        countN = countS = countE = countW = 0
        
        for i, ch in enumerate(s, start=1):
            if ch == 'N':
                countN += 1
            elif ch == 'S':
                countS += 1
            elif ch == 'E':
                countE += 1
            elif ch == 'W':
                countW += 1
                
            # For the prefix of length p = i:
            p = i
            # Horizontal: total moves and baseline contribution
            H = countE + countW
            base_x = abs(countE - countW)
            r_h = min(countE, countW)
            
            # Vertical: total moves and baseline
            V = countN + countS   # note: p = H + V
            base_y = abs(countN - countS)
            r_v = min(countN, countS)
            
            base = base_x + base_y
            
            # The effective modification budget cannot exceed p.
            mod = k if k < p else p
            
            # With each modification you can 'fix' a misaligned move (improve its contribution by 2).
            # But note that you cannot fix more than r_h+r_v misaligned moves.
            candidate = base + 2 * min(mod, r_h + r_v)
            if candidate > p:
                candidate = p
            if candidate > max_val:
                max_val = candidate
                
        return max_val

# For testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: s = "NWSE", k = 1. Expected output: 3.
    print(sol.maxDistance("NWSE", 1))
    # Example 2: s = "NSWWEW", k = 3. Expected output: 6.
    print(sol.maxDistance("NSWWEW", 3))
    # Some additional tests:
    print(sol.maxDistance("N", 0))    # Only one move, output: 1.
    print(sol.maxDistance("NNNN", 2)) # All moves already the same, output: 4.