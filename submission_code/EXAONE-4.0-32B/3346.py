class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = []
        rem = k
        for i in range(n):
            current_char = s[i]
            for c in "abcdefghijklmnopqrstuvwxyz":
                num1 = ord(current_char)
                num2 = ord(c)
                diff = abs(num1 - num2)
                cost_here = min(diff, 26 - diff)
                if cost_here <= rem:
                    t.append(c)
                    rem -= cost_here
                    break
        return ''.join(t)