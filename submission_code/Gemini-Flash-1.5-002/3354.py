class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        ans = list(s)
        used = set()
        for i in range(n):
            if ans[i] == '?':
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    if char not in used:
                        ans[i] = char
                        used.add(char)
                        break
        return "".join(ans)