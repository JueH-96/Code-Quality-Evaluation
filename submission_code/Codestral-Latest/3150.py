class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""

        for i in range(n):
            count = 0
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    length = j - i + 1
                    if length < min_len:
                        min_len = length
                        result = s[i:j+1]
                    elif length == min_len:
                        result = min(result, s[i:j+1])
                    break

        return result