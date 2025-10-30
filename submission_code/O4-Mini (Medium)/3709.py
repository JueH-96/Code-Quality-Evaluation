class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        # Slide a window of length k over the string
        for i in range(n - k + 1):
            c = s[i]
            # Check if all characters in s[i:i+k] are the same
            if all(s[j] == c for j in range(i, i + k)):
                # Check the character before the window (if any)
                if i > 0 and s[i - 1] == c:
                    continue
                # Check the character after the window (if any)
                if i + k < n and s[i + k] == c:
                    continue
                return True
        return False