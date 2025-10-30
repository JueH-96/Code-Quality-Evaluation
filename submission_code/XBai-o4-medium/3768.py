class Solution:
    def hasSameDigits(self, s: str) -> bool:
        while len(s) > 2:
            new_s = []
            for i in range(len(s) - 1):
                a = int(s[i])
                b = int(s[i+1])
                new_digit = (a + b) % 10
                new_s.append(str(new_digit))
            s = ''.join(new_s)
        return s[0] == s[1]