class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        max_len = 0
        len_s = len(s)
        len_t = len(t)
        for i in range(len_s + 1):
            for j in range(i, len_s + 1):
                len_s_sub = j - i
                for p in range(len_t + 1):
                    for q in range(p, len_t + 1):
                        len_t_sub = q - p
                        len_concat = len_s_sub + len_t_sub
                        is_pal = True
                        for k in range(len_concat // 2):
                            pos = k
                            pos_sym = len_concat - 1 - k
                            # Character at position pos in concat
                            if pos < len_s_sub:
                                char_pos = s[i + pos]
                            else:
                                offset_pos = pos - len_s_sub
                                char_pos = t[p + offset_pos]
                            # Character at symmetric position
                            if pos_sym < len_s_sub:
                                char_sym = s[i + pos_sym]
                            else:
                                offset_sym = pos_sym - len_s_sub
                                char_sym = t[p + offset_sym]
                            if char_pos != char_sym:
                                is_pal = False
                                break
                        if is_pal:
                            max_len = max(max_len, len_concat)
        return max_len