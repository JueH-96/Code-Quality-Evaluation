class Solution:
    def calculateScore(self, s: str) -> int:
        stacks = {}
        for c in "abcdefghijklmnopqrstuvwxyz":
            stacks[c] = []
        score = 0
        for i, c in enumerate(s):
            mirror_char = chr(ord('a') + ord('z') - ord(c))
            if stacks[mirror_char]:
                j = stacks[mirror_char].pop()
                score += (i - j)
            else:
                stacks[c].append(i)
        return score