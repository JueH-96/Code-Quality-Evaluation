class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            return min((ord(c1) - ord(c2)) % 26, (ord(c2) - ord(c1)) % 26)
        
        s = list(s)
        for i in range(len(s)):
            if k <= 0:
                break
            dist_to_a = distance(s[i], 'a')
            if dist_to_a <= k:
                s[i] = 'a'
                k -= dist_to_a
            else:
                s[i] = chr((ord(s[i]) - k) % 26 + ord('a'))
                k = 0
        return ''.join(s)