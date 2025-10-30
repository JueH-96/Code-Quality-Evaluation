from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        def mirror(c):
            return chr(2 * ord('a') + 25 - ord(c))
        
        stacks = defaultdict(list)
        score = 0
        for i in range(len(s)):
            c = s[i]
            m = mirror(c)
            if stacks[m]:
                j = stacks[m].pop()
                score += i - j
            else:
                stacks[c].append(i)
        return score