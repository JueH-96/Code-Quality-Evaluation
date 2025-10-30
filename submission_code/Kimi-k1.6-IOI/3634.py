from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        stacks = defaultdict(list)
        score = 0
        for i in range(len(s)):
            c = s[i]
            mirror_c = chr(219 - ord(c))
            if stacks[mirror_c]:
                j = stacks[mirror_c].pop()
                score += i - j
            else:
                stacks[c].append(i)
        return score