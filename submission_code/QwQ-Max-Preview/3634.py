class Solution:
    def calculateScore(self, s: str) -> int:
        from collections import defaultdict
        
        # Function to compute the mirror character
        def mirror_char(c):
            return chr(219 - ord(c))
        
        stacks = defaultdict(list)
        score = 0
        
        for i, c in enumerate(s):
            m = mirror_char(c)
            if stacks[m]:
                j = stacks[m].pop()
                score += i - j
            else:
                stacks[c].append(i)
        
        return score