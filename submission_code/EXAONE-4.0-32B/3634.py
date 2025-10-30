class Solution:
    def calculateScore(self, s: str) -> int:
        mirror_map = {}
        for i in range(26):
            c = chr(ord('a') + i)
            mirror_char = chr(ord('z') - i)
            mirror_map[c] = mirror_char
        
        stacks = {}
        for char in mirror_map:
            stacks[char] = []
        
        score = 0
        for i, char in enumerate(s):
            mirror_char = mirror_map[char]
            if stacks[mirror_char]:
                j = stacks[mirror_char].pop()
                score += (i - j)
            else:
                stacks[char].append(i)
        
        return score