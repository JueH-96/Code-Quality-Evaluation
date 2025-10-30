class Solution:
    def calculateScore(self, s: str) -> int:
        """
        We scan the string from left to right.
        For every character we keep a stack (actually a list) with the indices of
        yet-unmatched occurrences of this character.
        
        When we are at position i with character c
            – let mc be the mirror of c (index 25-idx if idx = ord(c)-97).
            – If the stack for mc is non–empty the top of that stack is the
              closest unmarked index j < i with the needed mirror character,
              because indices are pushed in increasing order.  
              We pop j, mark both positions and add (i-j) to the score.
            – Otherwise no partner exists, so i stays unmarked for now and we
              push i onto the stack of its own character.
              
        Time complexity:  O(n)
        Memory complexity: O(n) (worst case every position is kept on a stack)
        """
        
        # stacks[c] – list of unpaired indices whose character has code c
        stacks = [[] for _ in range(26)]
        score = 0
        
        for i, ch in enumerate(s):
            idx = ord(ch) - 97              # 0 … 25
            mirror_idx = 25 - idx           # mirror character index
            
            # If there is an available mirror to pair with
            if stacks[mirror_idx]:
                j = stacks[mirror_idx].pop()  # closest unmarked mirror index
                score += i - j                # add distance
            else:
                # No partner, keep this index for possible future pairing
                stacks[idx].append(i)
        
        return score