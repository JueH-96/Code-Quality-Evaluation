class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        for i in range(n - k + 1):
            sub = s[i:i+k]
            if len(set(sub)) == 1:
                valid = True
                if i > 0 and s[i-1] == sub[0]:
                    valid = False
                if i + k < n and s[i+k] == sub[0]:
                    valid = False
                if valid:
                    return True
        return False