class Solution:
    def minimumChairs(self, s: str) -> int:
        n = len(s)
        chairs = 0
        people = 0
        for i in range(n):
            if s[i] == 'E':
                people += 1
                if people > chairs:
                    chairs += 1
            elif s[i] == 'L':
                people -= 1
        return chairs