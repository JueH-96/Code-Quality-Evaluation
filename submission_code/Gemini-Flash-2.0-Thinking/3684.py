class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        n = len(s)
        len_prefix = len(prefix)
        len_suffix = len(suffix)

        for i in range(n):
            for j in range(i, n):
                substring = s[i : j + 1]

                if len(substring) >= len_prefix + len_suffix:
                    if substring.startswith(prefix) and substring.endswith(suffix):
                        return True

        return False