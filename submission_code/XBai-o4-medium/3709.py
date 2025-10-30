class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        for i in range(n - k + 1):
            sub = s[i:i+k]
            first_char = sub[0]
            all_same = True
            for c in sub:
                if c != first_char:
                    all_same = False
                    break
            if not all_same:
                continue
            # Check left and right conditions
            left_ok = True
            if i > 0:
                if s[i-1] == first_char:
                    left_ok = False
            right_ok = True
            if i + k < n:
                if s[i + k] == first_char:
                    right_ok = False
            if left_ok and right_ok:
                return True
        return False