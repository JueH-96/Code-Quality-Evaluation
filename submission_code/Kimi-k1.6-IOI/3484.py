class Solution:
    def getSmallestString(self, s: str) -> str:
        min_str = s
        n = len(s)
        for i in range(n - 1):
            a = int(s[i])
            b = int(s[i + 1])
            if a % 2 == b % 2:
                new_s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
                if new_s < min_str:
                    min_str = new_s
        return min_str