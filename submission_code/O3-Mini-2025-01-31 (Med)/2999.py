class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # The allowed operation swaps characters at positions that differ by 2.
        # In a 4-letter string, valid swaps are:
        # - swap indices 0 and 2 (even positions)
        # - swap indices 1 and 3 (odd positions)
        #
        # Therefore, we can only rearrange letters between even positions
        # and odd positions separately. To achieve equality, the multiset
        # of characters in the even positions of s1 must match those in s2,
        # and similarly for the odd positions.
        
        even_s1 = [s1[0], s1[2]]
        even_s2 = [s2[0], s2[2]]
        odd_s1 = [s1[1], s1[3]]
        odd_s2 = [s2[1], s2[3]]
        
        return sorted(even_s1) == sorted(even_s2) and sorted(odd_s1) == sorted(odd_s2)