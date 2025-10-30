class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_len = float('inf')
        lex_smallest_substring = ""

        for i in range(len(s)):
            for j in range(i, len(s)):
                sub = s[i:j+1]
                ones_count = sub.count('1')
                if ones_count == k:
                    if len(sub) < min_len:
                        min_len = len(sub)
                        lex_smallest_substring = sub
                    elif len(sub) == min_len:
                        if lex_smallest_substring == "" or sub < lex_smallest_substring:
                            lex_smallest_substring = sub

        return lex_smallest_substring