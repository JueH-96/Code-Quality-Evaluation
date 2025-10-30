class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Only swaps allowed are between indices with a difference of 2.
        # For a string of length 4, these swaps only affect:
        # - Even indices: positions 0 and 2.
        # - Odd indices: positions 1 and 3.
        #
        # We can only swap the characters in these independent groups.
        # Thus, s1 can be transformed into s2 if and only if:
        # - The multiset of characters at even positions in s1 is equal to
        #   the multiset at even positions in s2.
        # - The multiset of characters at odd positions in s1 is equal to
        #   the multiset at odd positions in s2.
        
        # Compare sorted characters in even and odd indices separately.
        return sorted(s1[0::2]) == sorted(s2[0::2]) and sorted(s1[1::2]) == sorted(s2[1::2])