class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        t = list(s)
        remaining_k = k
        for i in range(len(s)):
            original = s[i]
            for c in range(ord('a'), ord(original) + 1):
                candidate = chr(c)
                diff = abs(ord(original) - c)
                distance = min(diff, 26 - diff)
                if distance <= remaining_k:
                    t[i] = candidate
                    remaining_k -= distance
                    break
        return ''.join(t)