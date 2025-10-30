class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def char_distance(c1, c2):
            o1 = ord(c1) - ord('a')
            o2 = ord(c2) - ord('a')
            diff = abs(o1 - o2)
            return min(diff, 26 - diff)
        
        res = []
        remaining = k
        for c in s:
            for target in 'abcdefghijklmnopqrstuvwxyz':
                d = char_distance(c, target)
                if d <= remaining:
                    res.append(target)
                    remaining -= d
                    break
        return ''.join(res)