class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            return min(abs(ord(c1) - ord(c2)), 26 - abs(ord(c1) - ord(c2)))
        
        result = list(s)
        for i in range(len(s) - 1, -1, -1):
            if k >= distance(s[i], 'a'):
                k -= distance(s[i], 'a')
                result[i] = 'a'
            else:
                for j in range(26):
                    if distance(s[i], chr(ord('a') + j)) > k:
                        result[i] = chr(ord('a') + j - 1)
                        break
                break
        return "".join(result)