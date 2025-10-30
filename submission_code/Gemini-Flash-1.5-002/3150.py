class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_len = float('inf')
        result = ""
        for i in range(len(s)):
            count = 0
            for j in range(i, len(s)):
                if s[j] == '1':
                    count += 1
                if count == k:
                    substring = s[i:j+1]
                    if len(substring) < min_len:
                        min_len = len(substring)
                        result = substring
                    elif len(substring) == min_len:
                        result = min(result, substring)
                    break
        return result