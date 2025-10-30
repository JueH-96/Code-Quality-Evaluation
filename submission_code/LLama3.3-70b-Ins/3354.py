class Solution:
    def minimizeStringValue(self, s: str) -> str:
        result = []
        used = set()
        for char in s:
            if char == '?':
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c not in used:
                        result.append(c)
                        used.add(c)
                        break
            else:
                result.append(char)
                used.add(char)
        return ''.join(result)