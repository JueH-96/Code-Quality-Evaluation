from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        # Precompute the mirror for each character
        mirror = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            mirror[c] = chr(ord('z') - (ord(c) - ord('a')))
        
        # Use a dictionary to hold stacks for each character
        stacks = defaultdict(list)
        score = 0
        
        for i, c in enumerate(s):
            m = mirror[c]
            if stacks[m]:
                j = stacks[m].pop()
                score += (i - j)
            else:
                stacks[c].append(i)
        
        return score