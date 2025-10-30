class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        count = 1
        for i in range(1, n):
            if s[i] == s[i-1]:
                count += 1
            else:
                if count == k:
                    return True
                count = 1
        if count == k:
            return True
        return False