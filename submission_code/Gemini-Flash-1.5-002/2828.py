class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        ans = list(s)
        best = list(s)
        for i in range(n):
            for j in range(i, n):
                temp = list(s)
                for k in range(i, j + 1):
                    char_ord = ord(temp[k]) -1
                    if char_ord < ord('a'):
                        char_ord = ord('z')
                    temp[k] = chr(char_ord)
                if "".join(temp) < "".join(best):
                    best = temp
        return "".join(best)