from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        stack = defaultdict(list)
        score = 0
        for i, c in enumerate(s):
            m = chr(219 - ord(c))
            if stack[m]:
                j = stack[m].pop()
                score += i - j
            else:
                stack[c].append(i)
        return score