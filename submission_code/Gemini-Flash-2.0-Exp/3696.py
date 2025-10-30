class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                last_digit = int(sub[-1])
                if last_digit != 0 and int(sub) % last_digit == 0:
                    count += 1
        return count