class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        for i in range(n - k + 1):
            substr = s[i:i+k]
            if len(set(substr)) != 1:
                continue
            c = substr[0]
            before_ok = True
            if i > 0:
                if s[i-1] == c:
                    before_ok = False
            after_ok = True
            if i + k < n:
                if s[i + k] == c:
                    after_ok = False
            if before_ok and after_ok:
                return True
        return False