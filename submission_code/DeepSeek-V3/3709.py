class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        if k > n:
            return False
        for i in range(n - k + 1):
            substring = s[i:i+k]
            # Check if all characters in the substring are the same
            if len(set(substring)) == 1:
                char = substring[0]
                # Check the character before
                before_ok = True
                if i > 0:
                    before_char = s[i-1]
                    if before_char == char:
                        before_ok = False
                # Check the character after
                after_ok = True
                if i + k < n:
                    after_char = s[i+k]
                    if after_char == char:
                        after_ok = False
                if before_ok and after_ok:
                    return True
        return False