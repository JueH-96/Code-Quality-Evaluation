class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        if not prefix and not suffix:
            return True

        for i in range(len(s) + 1):
            if i + len(prefix) <= len(s) and i + len(prefix) + len(suffix) <= len(s) +1:
                if s.startswith(prefix, i) and s.endswith(suffix, i + len(prefix)):
                    return True
        return False