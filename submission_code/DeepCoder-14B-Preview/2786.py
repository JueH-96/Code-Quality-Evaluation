class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 1
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                count = 0
                for k in range(len(sub) - 1):
                    if sub[k] == sub[k+1]:
                        count += 1
                        if count > 1:
                            break
                if count <= 1:
                    current_len = j - i + 1
                    if current_len > max_len:
                        max_len = current_len
        return max_len