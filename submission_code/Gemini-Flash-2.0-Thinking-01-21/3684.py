class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index+1:]

        prefix_len = len(prefix)
        suffix_len = len(suffix)

        for i in range(len(s)):
            for j in range(i, len(s)):
                sub = s[i:j+1]
                sub_len = len(sub)

                if sub_len >= prefix_len + suffix_len:
                    if sub.startswith(prefix) and sub.endswith(suffix):
                        return True
                elif sub_len >= prefix_len and suffix_len == 0:
                    if sub.startswith(prefix):
                        return True
                elif sub_len >= suffix_len and prefix_len == 0:
                    if sub.endswith(suffix):
                        return True
                elif prefix_len == 0 and suffix_len == 0:
                    return True

        return False