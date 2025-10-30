class Solution:
    def maximumLength(self, s: str) -> int:
        max_len = -1
        i = 0
        while i < len(s):
            count = 1
            j = i + 1
            while j < len(s) and s[j] == s[i]:
                count += 1
                j += 1
            k = i + 2 * count
            while k < len(s) and s[i:k].count(s[i]) >= 3:
                max_len = max(max_len, count)
                k += 1
            i = j
        return max_len