class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        candidates = [s]
        for i in range(n - 1):
            a = int(s[i])
            b = int(s[i+1])
            if (a % 2) == (b % 2):
                new_str = s[:i] + s[i+1] + s[i] + s[i+2:]
                candidates.append(new_str)
        return min(candidates)