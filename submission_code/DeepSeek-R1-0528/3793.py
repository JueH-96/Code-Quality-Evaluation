class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        ans = 0
        for i in range(n + 1):
            for j in range(i, n + 1):
                s_len = j - i
                for k in range(m + 1):
                    for l in range(k, m + 1):
                        t_len = l - k
                        total_len = s_len + t_len
                        if total_len == 0:
                            continue
                        left_ptr = 0
                        right_ptr = total_len - 1
                        valid = True
                        while left_ptr < right_ptr:
                            if left_ptr < s_len:
                                left_char = s[i + left_ptr]
                            else:
                                left_char = t[k + (left_ptr - s_len)]
                            if right_ptr < s_len:
                                right_char = s[i + right_ptr]
                            else:
                                right_char = t[k + (right_ptr - s_len)]
                            if left_char != right_char:
                                valid = False
                                break
                            left_ptr += 1
                            right_ptr -= 1
                        if valid:
                            ans = max(ans, total_len)
        return ans