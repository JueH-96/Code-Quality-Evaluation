class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        # Iterate over all possible starting indices for a substring of length k
        for i in range(n - k + 1):
            # Check if the substring s[i:i+k] is made of the same character
            c = s[i]
            if all(ch == c for ch in s[i:i+k]):
                # Check the character before the substring, if it exists
                if i > 0 and s[i-1] == c:
                    continue
                # Check the character after the substring, if it exists
                if i + k < n and s[i+k] == c:
                    continue
                # Found a valid special substring
                return True
        return False