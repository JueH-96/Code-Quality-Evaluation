class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # By carefully analyzing the operations, one can see
        # that any character can be repeatedly chosen to remove
        # pairs of the same character around it. However, the
        # chosen occurrence itself never gets removed.
        #
        # Therefore, any character can be reduced to at most
        # one occurrence in the final string. The minimal
        # length is just the number of distinct characters in s.
        
        return len(set(s))