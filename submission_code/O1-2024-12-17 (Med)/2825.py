class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # The key insight is that each character in the string can be reduced
        # to at most one occurrence, since picking a middle occurrence of a
        # repeated character allows us to remove the closest matching
        # characters on both sides. Thus, the minimal length of the string
        # after performing all possible operations is simply the number of
        # distinct characters in the string.
        
        return len(set(s))