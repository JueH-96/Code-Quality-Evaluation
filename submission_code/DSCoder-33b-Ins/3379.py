class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0
        depth = 0
        for i in range(len(s)):
            if s[i] == 'a':
                depth += 1
            else:
                while depth > 0:
                    score += depth
                    depth -= 1
        while depth > 0:
            score += depth
            depth -= 1
        return score