class Solution:
    def calculateScore(self, s: str) -> int:
        # We'll map each character 'a'..'z' to an index 0..25
        # and store the unmarked positions in a stack for each character.
        # The mirror of a letter with index i is (25 - i).
        # Example: index('a') = 0, mirror = 25 -> 'z'
        #          index('b') = 1, mirror = 24 -> 'y'
        
        # Prepare 26 stacks (one for each letter).
        stacks = [[] for _ in range(26)]
        
        def char_to_index(ch):
            return ord(ch) - ord('a')
        
        score = 0
        for i, ch in enumerate(s):
            idx = char_to_index(ch)
            mirror_idx = 25 - idx  # index of the mirror letter
            if stacks[mirror_idx]:
                # Found a matching (unmarked) index j for which s[j] is mirror of ch
                j = stacks[mirror_idx].pop()
                score += (i - j)
                # Do not push i, because i is now used
            else:
                # No match found, so i remains unmarked for now
                stacks[idx].append(i)
        
        return score