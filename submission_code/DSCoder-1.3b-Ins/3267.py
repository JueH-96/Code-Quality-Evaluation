class Solution:
    def maximumLength(self, s: str) -> int:
        max_len = 0
        cur_len = 1
        special_substr = set()
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur_len += 1
            else:
                if s[i] not in special_substr:
                    special_substr.add(s[i-1])
                    cur_len = 2
                else:
                    cur_len = 1
            max_len = max(max_len, cur_len)
        return max_len if max_len >= 3 else -1