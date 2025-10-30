class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = n + 1
        min_str = ""
        for i in range(n):
            if s[i] == '0':
                continue
            count = 1
            for j in range(i + 1, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    if j - i + 1 < min_len:
                        min_len = j - i + 1
                        min_str = s[i:j+1]
                    break
        return min_str