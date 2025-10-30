class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        swapped = False
        for i in range(n - 1):
            if (int(s[i]) % 2 == 0 and int(s[i+1]) % 2 == 0) or \
               (int(s[i]) % 2 != 0 and int(s[i+1]) % 2 != 0):
                if s[i] > s[i+1]:
                    s[i], s[i+1] = s[i+1], s[i]
                    swapped = True

        return "".join(s)