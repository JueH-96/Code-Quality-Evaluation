from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        stacks = defaultdict(list)
        score = 0
        for i, c in enumerate(s):
            # Calculate the mirror character
            mirror = chr(219 - ord(c))
            # Check if there's a valid j in the mirror's stack
            if stacks[mirror]:
                j = stacks[mirror].pop()
                score += i - j
            else:
                # Push current index to its character's stack
                stacks[c].append(i)
        return score