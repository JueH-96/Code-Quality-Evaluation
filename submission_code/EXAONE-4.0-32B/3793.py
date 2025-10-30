class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        best = 0
        for i in range(n):
            for j in range(i, n + 1):
                lenA = j - i
                if lenA + m <= best:
                    continue
                for k in range(m):
                    if lenA + (m - k) <= best:
                        break
                    for l in range(m, k - 1, -1):
                        lenB = l - k
                        total_len = lenA + lenB
                        if total_len <= best:
                            break
                        left_ptr = 0
                        right_ptr = total_len - 1
                        is_pal = True
                        while left_ptr < right_ptr:
                            if left_ptr < lenA:
                                char_left = s[i + left_ptr]
                            else:
                                char_left = t[k + (left_ptr - lenA)]
                            if right_ptr < lenA:
                                char_right = s[i + right_ptr]
                            else:
                                char_right = t[k + (right_ptr - lenA)]
                            if char_left != char_right:
                                is_pal = False
                                break
                            left_ptr += 1
                            right_ptr -= 1
                        if is_pal:
                            best = total_len
                            break
        return best