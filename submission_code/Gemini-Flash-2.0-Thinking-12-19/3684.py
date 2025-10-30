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
                if len(sub) >= prefix_len + suffix_len:
                    if sub[:prefix_len] == prefix and sub[len(sub)-suffix_len:] == suffix:
                        return True
        return False