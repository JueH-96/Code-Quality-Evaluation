class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if (int(s[i]) % 2 == int(s[j]) % 2) and s[j] < s[i]:
                    s[i], s[j] = s[j], s[i]
                    return "".join(s)
        return "".join(s)