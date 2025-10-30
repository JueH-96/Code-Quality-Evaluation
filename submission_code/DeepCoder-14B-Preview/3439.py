class Solution:
    def getSmallestString(self, s: str) -> str:
        candidates = [s]
        for i in range(len(s) - 1):
            if (int(s[i]) % 2) == (int(s[i+1]) % 2):
                new_s = s[:i] + s[i+1] + s[i] + s[i+2:]
                candidates.append(new_s)
        return min(candidates)