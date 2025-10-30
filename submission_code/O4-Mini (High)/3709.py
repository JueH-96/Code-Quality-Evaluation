class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        for i in range(n - k + 1):
            c = s[i]
            # Check if s[i:i+k] are all the same character c
            if all(s[j] == c for j in range(i, i + k)):
                # Check left boundary
                if i > 0 and s[i - 1] == c:
                    continue
                # Check right boundary
                if i + k < n and s[i + k] == c:
                    continue
                return True
        return False