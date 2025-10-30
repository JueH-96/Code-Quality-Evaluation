class Solution:
    def findValidPair(self, s: str) -> str:
        for i in range(len(s) - 1):
            if s[i] != s[i+1]:
                d1 = s[i]
                d2 = s[i+1]
                count1 = s.count(d1)
                count2 = s.count(d2)
                if count1 == int(d1) and count2 == int(d2):
                    return d1 + d2
        return ""