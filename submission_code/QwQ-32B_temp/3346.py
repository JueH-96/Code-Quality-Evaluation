class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        remaining = k
        for char in s:
            for candidate in 'abcdefghijklmnopqrstuvwxyz':
                d = self.distance(char, candidate)
                if d <= remaining:
                    res.append(candidate)
                    remaining -= d
                    break
        return ''.join(res)
    
    def distance(self, a: str, b: str) -> int:
        x = ord(a) - ord('a')
        y = ord(b) - ord('a')
        return min(abs(x - y), 26 - abs(x - y))