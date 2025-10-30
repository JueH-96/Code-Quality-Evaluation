class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        res = 0
        left, right = -1, -1
        for i in range(len(s)):
            if s[i] == c:
                if left == -1:
                    left = i
                right = i
                res += left + 1
        return res + right - len(s) + 1 if right != -1 else 0