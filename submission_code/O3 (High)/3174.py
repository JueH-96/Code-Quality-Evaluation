class Solution:
    def minChanges(self, s: str) -> int:
        """
        A string is beautiful iff every adjacent pair (positions 2k and 2k+1)
        consists of identical characters.  
        Proof sketch:
          • Every required substring has even length, so the end index of every
            substring (cut position) must have odd parity (since the first cut
            is at index -1).  
          • Therefore a cut can never be placed between an even index (2k) and
            the next odd index (2k+1).  
          • Consequently characters s[2k] and s[2k+1] must belong to the same
            substring and hence must be equal.  

        Conversely, if each pair is equal, the string can be partitioned into
        length-2 blocks ("00" or "11"), or neighbouring blocks with the same
        character can be merged, so a valid partition always exists.

        Hence the minimum number of changes equals the number of pairs with
        different characters.
        """
        changes = 0
        # step through the string pair-wise
        for i in range(0, len(s), 2):
            if s[i] != s[i + 1]:
                changes += 1
        return changes