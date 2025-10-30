class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.index("*")
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        if not prefix and not suffix:
            return True

        for i in range(len(s) - len(prefix) - len(suffix) + 1):
            if s[i:i + len(prefix)] == prefix:
                if s[i + len(prefix):].endswith(suffix):
                    return True
        return False