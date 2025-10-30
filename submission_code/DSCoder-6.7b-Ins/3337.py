class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = len(s)
        count = 0
        c_indices = [i for i in range(n) if s[i] == c]
        for i in range(n):
            if s[i] == c:
                left = i
                right = i
                while left >= 0 and right < n and s[left] == s[right] == c:
                    count += 1
                    left -= 1
                    right += 1
        return count