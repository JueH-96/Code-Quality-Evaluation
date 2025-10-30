class Solution:
    def calculateScore(self, s: str) -> int:
        mirror = {chr(i): chr(ord('z') - (i - ord('a'))) for i in range(ord('a'), ord('z') + 1)}
        from collections import defaultdict

        char_map = defaultdict(list)
        total_score = 0

        for i, c in enumerate(s):
            m = mirror[c]
            if char_map[m]:
                j = char_map[m].pop()
                total_score += i - j
            else:
                char_map[c].append(i)
        return total_score