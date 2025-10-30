class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        ans = ""
        for i in range(n):
            for j in range(26):
                char = chr(ord('a') + j)
                dist = abs(ord(s[i]) - ord(char))
                dist = min(dist, 26 - dist)
                if dist <= k:
                    ans += char
                    k -= dist
                    break
        return ans