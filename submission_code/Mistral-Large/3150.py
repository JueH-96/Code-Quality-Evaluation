class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""

        # Generate all possible substrings
        for i in range(n):
            count = 0
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    substring = s[i:j+1]
                    if len(substring) < min_len:
                        min_len = len(substring)
                        result = substring
                    elif len(substring) == min_len:
                        result = min(result, substring)

        return result