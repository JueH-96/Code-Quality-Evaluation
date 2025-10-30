class Solution:
    def getSmallestString(self, s: str) -> str:
        min_s = s
        for i in range(len(s) - 1):
            a, b = int(s[i]), int(s[i+1])
            if a % 2 == b % 2:
                # Swap the adjacent digits
                swapped = s[:i] + s[i+1] + s[i] + s[i+2:]
                if swapped < min_s:
                    min_s = swapped
        return min_s